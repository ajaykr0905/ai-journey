import itertools

import numpy
import pytest

from ai_journey.gpu_reduce_benchmark import (
    DECLARED_TOLERANCE,
    ORDER_BOUND,
    benchmark_sweep,
    rates,
)
from ai_journey.gpu_reduce_bound import (
    VENDOR_DEPTH,
    benchmark_values,
    exact_sums,
    partial_count,
    summation_depth,
)
from ai_journey.gpu_reduce_runtime import SIDES, ReduceWorkload
from ai_journey.gpu_reduce_source import R0, R4, STEP_KERNELS, GpuReduceError
from ai_journey.kernel_metal_benchmark import DeviceTiming, RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from test_gpu_reduce_correctness import StandInBackend
from test_gpu_reduce_runtime import reduce_runtime
from test_kernel_metal_runtime import requires_metal

SIZES = (4096, 70001)


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def sweep(backend, sizes=SIZES, clock=None, budget=None, repeats=7):
    return benchmark_sweep(
        backend,
        seed=78,
        sizes=sizes,
        warmup=1,
        repeats=repeats,
        target_bytes_per_sample=1 << 20,
        clock=clock or scripted_clock(),
        budget=budget,
    )


class RecordingBackend(StandInBackend):
    """A stand-in whose side calls log their names, so the order shows."""

    def __init__(self, **options):
        super().__init__(**options)
        self.calls = []

    def workload(self, source, n):
        inner = super().workload(source, n)

        def call(name):
            return lambda: self.calls.append(name)

        return ReduceWorkload(
            sides={name: call(name) for name in SIDES},
            synchronize=inner.synchronize,
            check=inner.check,
        )


def test_every_side_is_timed_on_one_upload_and_checked_against_its_prefix():
    backend = StandInBackend()

    points = sweep(backend)
    values = benchmark_values(78, max(SIZES))

    assert backend.uploads == [max(SIZES)] and backend.workloads == list(SIZES)
    for point, n in zip(points, SIZES):
        mapping = point.to_mapping()
        assert mapping["n"] == n and mapping["bytes_per_call"] == 4 * n
        assert mapping["side_order"] == list(SIDES)
        assert list(mapping["launches"]) == list(STEP_KERNELS)
        for name in STEP_KERNELS:
            assert mapping["launches"][name]["groups"] == [partial_count(name, n)]
        sums = exact_sums(values[:n])
        assert mapping["reference"] == sums.reference
        assert mapping["sum_abs"] == sums.sum_abs
        for name, check in mapping["checks"].items():
            vendor = name == "torch_sum"
            assert check["rule"] == (DECLARED_TOLERANCE if vendor else ORDER_BOUND)
            assert check["depth"] == (
                VENDOR_DEPTH if vendor else summation_depth(name, n)
            )
            assert check["same_bits_on_repeat"] is True
            assert check["bound_ratio"] <= 1.0
    assert [point.timings[0].calls_per_sample for point in points] == [64, 4]


def test_the_side_order_moves_by_one_side_at_every_repeat():
    backend = RecordingBackend()

    sweep(backend, sizes=(4096,), repeats=7)

    batch = 64
    warm, timed = backend.calls[: 7 * batch], backend.calls[7 * batch :]
    assert warm == [name for name in SIDES for _ in range(batch)]
    starts = [timed[repeat * 7 * batch] for repeat in range(7)]
    assert starts == [SIDES[repeat % 7] for repeat in range(7)]
    assert all(timed.count(name) == 7 * batch for name in SIDES)


def test_rates_divide_the_bytes_of_a_sample_by_its_median_and_quartiles():
    timing = DeviceTiming("x", 4, 1, 4, (400, 100, 300, 200), "injected")

    entry = rates(timing, 1000)

    assert entry["bytes_per_sample"] == 4000
    assert entry["timing"]["median_ns"] == 200
    assert entry["median_gb_per_second"] == 4000 / 200
    assert entry["best_gb_per_second"] == 4000 / 100
    assert entry["iqr_gb_per_second"] == [4000 / 300, 4000 / 100]
    assert entry["median_ns_per_call"] == 50.0


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (StandInBackend(scale={R4: 1e4}), "reduce_r4_simd_sum sits"),
        (StandInBackend(scale={"torch_sum": 1.5}), "declared tolerance is"),
    ],
)
def test_a_side_outside_its_bound_or_tolerance_stops_the_run(backend, message):
    with pytest.raises(GpuReduceError, match=message):
        sweep(backend)


def test_a_workload_in_another_order_or_a_spent_budget_stops_the_run():
    class Reordered(StandInBackend):
        def workload(self, source, n):
            inner = super().workload(source, n)
            return ReduceWorkload(
                sides=dict(reversed(list(inner.sides.items()))),
                synchronize=inner.synchronize,
                check=inner.check,
            )

    with pytest.raises(GpuReduceError, match="R0 to R5 and then torch.sum"):
        sweep(Reordered())
    clock = scripted_clock(step=10**9)
    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        sweep(StandInBackend(), clock=clock, budget=RunBudget(1, clock))
    with pytest.raises(GpuReduceError, match="1 to 16 sizes"):
        sweep(StandInBackend(), sizes=())


@requires_metal
def test_a_real_sweep_times_every_side_and_checks_each_result():
    points = benchmark_sweep(
        reduce_runtime(),
        seed=78,
        sizes=(1 << 16, 70001),
        warmup=1,
        repeats=7,
        target_bytes_per_sample=1 << 22,
    )

    for point in points:
        mapping = point.to_mapping()
        for name in SIDES:
            assert mapping["sides"][name]["median_gb_per_second"] > 0
            assert mapping["checks"][name]["bound_ratio"] <= 1.0
        for name in STEP_KERNELS:
            assert mapping["checks"][name]["same_bits_on_repeat"] is True
        assert point.timings[0].timer == "time.perf_counter_ns"
        assert numpy.float32(point.checks[R0].check.output).dtype == numpy.float32
