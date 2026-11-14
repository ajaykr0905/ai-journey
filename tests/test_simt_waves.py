from fractions import Fraction

import pytest

from ai_journey.simt_device import PINNED_DEVICES, SimtError
from ai_journey.simt_waves import WaveQuantization, wave_quantization

A100 = PINNED_DEVICES["a100-sxm4-80gb"]
V100 = PINNED_DEVICES["tesla-v100-sxm2"]
T4 = PINNED_DEVICES["tesla-t4"]


def test_fixed_grid_quantizes_differently_on_each_device():
    # 1,024 blocks at 4 resident blocks per SM:
    # A100: 108 * 4 = 432 slots -> 2 full waves + 160 tail blocks -> 3 waves,
    #       1,024 / 1,296 slots = 79.0% utilization, last wave 160 / 432 full.
    # V100:  80 * 4 = 320 slots -> 3 full + 64 -> 4 waves, 1,024 / 1,280 = 80%.
    # T4:    40 * 4 = 160 slots -> 6 full + 64 -> 7 waves, 1,024 / 1,120 = 91.4%.
    expected = {
        A100: (432, 2, 160, 3, Fraction(160, 432), Fraction(1_024, 1_296)),
        V100: (320, 3, 64, 4, Fraction(1, 5), Fraction(4, 5)),
        T4: (160, 6, 64, 7, Fraction(2, 5), Fraction(1_024, 1_120)),
    }
    for device, values in expected.items():
        slots, full, tail, waves, fill, utilization = values
        result = wave_quantization(device, total_blocks=1_024, active_blocks_per_sm=4)
        assert result.slots_per_wave == slots
        assert result.full_waves == full
        assert result.tail_blocks == tail
        assert result.waves == waves
        assert result.last_wave_fill == fill
        assert result.sm_slot_utilization == utilization


def test_exact_multiples_and_sub_wave_grids():
    # 864 = 108 * 8 blocks exactly fill one A100 wave at 8 blocks per SM.
    exact = wave_quantization(A100, total_blocks=864, active_blocks_per_sm=8)
    assert (exact.waves, exact.tail_blocks, exact.last_wave_fill) == (1, 0, 1)
    assert exact.sm_slot_utilization == 1
    # 100 blocks never fill the first wave: 100 / 864 of the slots are used.
    small = wave_quantization(A100, total_blocks=100, active_blocks_per_sm=8)
    assert (small.full_waves, small.waves) == (0, 1)
    assert small.sm_slot_utilization == Fraction(100, 864)
    # The 2^24-element vector add grid: 65,536 blocks over 864 slots is 75 full
    # waves plus 736 blocks, 76 waves, 65,536 / 65,664 = 99.8% utilization.
    vector_add = wave_quantization(A100, total_blocks=65_536, active_blocks_per_sm=8)
    assert (vector_add.full_waves, vector_add.tail_blocks) == (75, 736)
    assert vector_add.sm_slot_utilization == Fraction(65_536, 65_664)
    assert vector_add.to_mapping()["waves"] == 76


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda: wave_quantization(A100, total_blocks=0, active_blocks_per_sm=1),
            SimtError,
            "total_blocks",
        ),
        (
            lambda: wave_quantization(T4, total_blocks=1, active_blocks_per_sm=17),
            SimtError,
            "active_blocks_per_sm must be between 1 and 16",
        ),
        (
            lambda: wave_quantization(A100, total_blocks=1, active_blocks_per_sm=True),
            TypeError,
            "active_blocks_per_sm must be an integer",
        ),
        (
            lambda: wave_quantization("a100", total_blocks=1, active_blocks_per_sm=1),
            TypeError,
            "SimtDevice",
        ),
        (lambda: WaveQuantization("a100", 10, 0, 1), SimtError, "sm_count"),
    ],
)
def test_wave_inputs_are_strict(call, error, message):
    with pytest.raises(error, match=message):
        call()
