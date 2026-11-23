import itertools
import time

import numpy
import pytest

from ai_journey.gpu_scan_benchmark import (
    benchmark_point,
    benchmark_sweep,
    compulsory_bytes,
    copy_bytes,
    modeled_bytes,
)
from ai_journey.gpu_scan_plan import COPY, VENDOR
from ai_journey.gpu_scan_reference import error_bound
from ai_journey.gpu_scan_runtime import ScanWorkload
from ai_journey.gpu_scan_source import (
    KERNELS,
    S0_HILLIS_STEELE,
    S1_BLELLOCH,
    S2_SIMD,
    GpuScanError,
)
from test_gpu_scan_correctness import StandInBackend
from test_gpu_scan_runtime import scan_runtime
from test_kernel_metal_runtime import requires_metal


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def run_point(backend, length=5003, names=tuple(KERNELS), clock=None):
    return benchmark_point(
        backend,
        seed=78,
        index=0,
        length=length,
        names=names,
        warmup=1,
        repeats=3,
        target_bytes_per_sample=1 << 16,
        clock=scripted_clock() if clock is None else clock,
    )


def test_the_traffic_model_counts_every_level_and_every_add():
    assert compulsory_bytes(1 << 20) == 8 << 20
    assert modeled_bytes(1000) == 4 * (2 * 1000 + 1)
    assert modeled_bytes(1 << 20) == 4 * (
        (2 * (1 << 20) + 1024) + (2 * 1024 + 1) + (2 * ((1 << 20) - 1024) + 1023)
    )
    assert modeled_bytes(1 << 20) == 16785408
    assert 1.99 < modeled_bytes(1 << 24) / compulsory_bytes(1 << 24) < 2.01
    assert copy_bytes(5) == 64 and copy_bytes(8) == 64 and copy_bytes(9) == 96


def test_a_point_times_every_side_in_turn_on_one_input():
    point = run_point(StandInBackend()).to_mapping()

    assert point["side_order"] == [*KERNELS, VENDOR, COPY]
    assert (point["levels"], point["dispatches"]) == ([5003, 5], 3)
    assert point["bytes_per_call"] == 8 * 5003
    assert point["copy_bytes_per_call"] == 32 * 1251
    assert point["modeled_bytes_per_call"] == modeled_bytes(5003)
    assert list(point["launches"]) == list(KERNELS)
    assert point["copy_launch"]["extent"] == [313]
    for name, side in point["sides"].items():
        moved = 2 * (point["copy_bytes_per_call"] if name == COPY else 8 * 5003)
        assert side["calls_per_sample"] == 2 and side["warmup"] == 1
        assert side["samples_ns"] == [1000, 1000, 1000]
        assert side["bytes_per_sample"] == moved
        assert side["median_gb_per_second"] == moved / 1000
        assert side["iqr_gb_per_second"] == [moved / 1000, moved / 1000]
    checks = point["checks"]
    assert checks[S0_HILLIS_STEELE]["same_bits_as_s0"] is True
    assert checks[S1_BLELLOCH]["same_bits_as_s0"] is False
    assert checks[S2_SIMD]["bound"] == error_bound(S2_SIMD, 5003)
    assert checks[VENDOR]["bound"] is None and checks[VENDOR]["bound_ratio"] is None
    assert checks[COPY] == {"same_bits_as_input": True}
    assert all(
        timing.timer == "injected" for timing in run_point(StandInBackend()).timings
    )


def test_a_timed_output_above_its_bound_or_a_changed_copy_stops_the_point():
    with pytest.raises(GpuScanError, match="above its bound"):
        run_point(StandInBackend(scale={S1_BLELLOCH: 1.01}))

    class ChangedCopy(StandInBackend):
        def workload(self, values, names):
            workload = super().workload(values, names)

            def check():
                outputs = dict(workload.check())
                outputs[COPY] = outputs[COPY] + numpy.float32(1)
                return outputs

            return ScanWorkload(workload.sides, workload.synchronize, check)

    with pytest.raises(GpuScanError, match="changed its input"):
        run_point(ChangedCopy())


def test_a_refused_kernel_is_neither_bound_nor_timed():
    point = run_point(StandInBackend(widths={S2_SIMD: 16})).to_mapping()

    assert point["side_order"] == [S0_HILLIS_STEELE, S1_BLELLOCH, VENDOR, COPY]
    assert list(point["launches"]) == [S0_HILLIS_STEELE, S1_BLELLOCH]
    assert S2_SIMD not in point["checks"]


def test_a_workload_in_another_order_is_refused():
    class Shuffled(StandInBackend):
        def workload(self, values, names):
            workload = super().workload(values, names)
            sides = dict(reversed(list(workload.sides.items())))
            return ScanWorkload(sides, workload.synchronize, workload.check)

    with pytest.raises(GpuScanError, match="then torch.cumsum, then the copy"):
        run_point(Shuffled())
    for lengths in ((), tuple(range(1000, 1017))):
        with pytest.raises(GpuScanError, match="1 to 16 lengths"):
            benchmark_sweep(
                StandInBackend(),
                seed=78,
                lengths=lengths,
                names=tuple(KERNELS),
                warmup=0,
                repeats=3,
                target_bytes_per_sample=1 << 16,
                clock=scripted_clock(),
            )


@requires_metal
def test_this_gpu_times_a_short_point_and_passes_every_check():
    point = run_point(scan_runtime(), clock=time.perf_counter_ns).to_mapping()

    assert point["side_order"] == [*KERNELS, VENDOR, COPY]
    for name in KERNELS:
        assert point["checks"][name]["bound_ratio"] <= 1.0, name
    assert point["checks"][S0_HILLIS_STEELE]["same_bits_as_s0"] is True
    assert point["checks"][COPY] == {"same_bits_as_input": True}
    assert all(len(side["samples_ns"]) == 3 for side in point["sides"].values())
