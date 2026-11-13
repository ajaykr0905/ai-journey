from fractions import Fraction

import pytest

from ai_journey.simt_device import PINNED_DEVICES, SimtError
from ai_journey.simt_latency import MemoryAccessPattern, memory_latency_hiding

A100 = PINNED_DEVICES["a100-sxm4-80gb"]
V100 = PINNED_DEVICES["tesla-v100-sxm2"]
T4 = PINNED_DEVICES["tesla-t4"]
SCALAR = MemoryAccessPattern(bytes_per_lane=4, active_lanes=32, independent_loads=1)


def test_a100_bytes_in_flight_follow_littles_law():
    # Per SM: 2,039e9 B/s / (108 SMs * 1.41e9 cycles/s) = 50,975 / 3,807
    # = 13.3898 bytes per cycle. At a declared 400 cycles: 5,355.92 bytes.
    # One scalar float load per lane keeps 4 * 32 = 128 bytes in flight per
    # warp, so ceil(5,355.92 / 128) = 42 warps are needed.
    result = memory_latency_hiding(
        A100, latency_cycles=400, pattern=SCALAR, resident_warps=64
    )

    assert result.bytes_per_cycle_per_sm == Fraction(50_975, 3_807)
    assert result.bytes_in_flight_per_sm == Fraction(50_975 * 400, 3_807)
    assert result.latency_seconds == Fraction(400, 1_410_000_000)
    assert result.warps_needed == 42
    assert result.full_bandwidth_within_warp_limit is True
    assert result.sustainable_bandwidth_fraction == 1


def test_longer_latency_can_exceed_the_warp_limit_without_more_bytes_per_warp():
    # At a declared 800 cycles the SM needs 10,711.85 bytes in flight:
    # ceil(10,711.85 / 128) = 84 warps, more than the 64 an A100 SM can hold.
    # 64 * 128 = 8,192 bytes in flight sustain 8,192 / 10,711.85 = 76.48%.
    scalar = memory_latency_hiding(
        A100, latency_cycles=800, pattern=SCALAR, resident_warps=64
    )
    assert scalar.warps_needed == 84
    assert scalar.full_bandwidth_within_warp_limit is False
    assert scalar.sustainable_bandwidth_fraction == Fraction(64 * 128) / Fraction(
        50_975 * 800, 3_807
    )
    assert round(float(scalar.sustainable_bandwidth_fraction), 4) == 0.7648
    # 16-byte vector loads put 512 bytes per warp in flight: 21 warps suffice.
    vector = memory_latency_hiding(
        A100,
        latency_cycles=800,
        pattern=MemoryAccessPattern(16, 32, 1),
        resident_warps=64,
    )
    assert vector.warps_needed == 21
    assert vector.sustainable_bandwidth_fraction == 1
    # Two independent float loads per lane (a vector add) halve the warps.
    paired = memory_latency_hiding(
        A100,
        latency_cycles=800,
        pattern=MemoryAccessPattern(4, 32, 2),
        resident_warps=16,
    )
    assert paired.warps_needed == 42
    assert paired.sustainable_bandwidth_fraction == Fraction(16 * 256) / Fraction(
        50_975 * 800, 3_807
    )


def test_bandwidth_per_sm_differs_across_declared_devices():
    # V100: 900e9 / (80 * 1.53e9) = 125/17 = 7.35 B/cycle -> 5,882.35 bytes at
    # 800 cycles -> 46 scalar warps. T4: 300e9 / (40 * 1.59e9) = 250/53 =
    # 4.72 B/cycle -> 3,773.58 bytes -> 30 warps, within its 32-warp limit.
    v100 = memory_latency_hiding(
        V100, latency_cycles=800, pattern=SCALAR, resident_warps=64
    )
    t4 = memory_latency_hiding(
        T4, latency_cycles=800, pattern=SCALAR, resident_warps=32
    )

    assert v100.bytes_per_cycle_per_sm == Fraction(125, 17)
    assert v100.warps_needed == 46
    assert t4.bytes_per_cycle_per_sm == Fraction(250, 53)
    assert t4.warps_needed == 30
    assert t4.full_bandwidth_within_warp_limit is True
    # Eight resident T4 warps sustain 8 * 128 / 3,773.58 = 27.1% of 300 GB/s.
    low = memory_latency_hiding(
        T4, latency_cycles=800, pattern=SCALAR, resident_warps=8
    )
    assert low.sustainable_bandwidth_fraction == Fraction(1_024 * 53, 250 * 800)
    assert low.sustainable_bandwidth_bytes_per_second == (
        Fraction(1_024 * 53, 250 * 800) * 300 * 10**9
    )


def test_partial_lanes_reduce_bytes_in_flight_per_warp():
    half = MemoryAccessPattern(bytes_per_lane=4, active_lanes=16, independent_loads=1)
    result = memory_latency_hiding(
        A100, latency_cycles=400, pattern=half, resident_warps=64
    )

    assert half.bytes_per_warp == 64
    assert result.warps_needed == 84
    mapping = result.to_mapping()
    assert mapping["bytes_per_warp"] == 64
    assert mapping["access"] == {
        "bytes_per_lane": 4,
        "active_lanes": 16,
        "independent_loads": 1,
    }
    assert mapping["full_bandwidth_within_warp_limit"] is False


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: MemoryAccessPattern(3, 32, 1), SimtError, "1, 2, 4, 8, or 16"),
        (lambda: MemoryAccessPattern(4, 0, 1), SimtError, "active_lanes"),
        (lambda: MemoryAccessPattern(4, 32, True), TypeError, "independent_loads"),
        (
            lambda: memory_latency_hiding(
                T4, latency_cycles=400, pattern=SCALAR, resident_warps=33
            ),
            SimtError,
            "resident_warps must be between 1 and 32",
        ),
        (
            lambda: memory_latency_hiding(
                A100, latency_cycles=0, pattern=SCALAR, resident_warps=1
            ),
            SimtError,
            "latency_cycles",
        ),
        (
            lambda: memory_latency_hiding(
                A100,
                latency_cycles=400,
                pattern=MemoryAccessPattern(4, 64, 1),
                resident_warps=1,
            ),
            SimtError,
            "cannot exceed the warp size",
        ),
        (
            lambda: memory_latency_hiding(
                A100.architecture, latency_cycles=400, pattern=SCALAR, resident_warps=1
            ),
            TypeError,
            "SimtDevice",
        ),
    ],
)
def test_memory_latency_inputs_are_strict(call, error, message):
    with pytest.raises(error, match=message):
        call()
