import itertools
import time

import pytest

from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_benchmark import (
    MatmulPoint,
    agreement_bound,
    benchmark_sweep,
    calls_per_sample,
    ceiling_summary,
    time_sides,
)
from ai_journey.matmul_metal_mapping import MatmulMetalError, matmul_shape
from ai_journey.matmul_metal_reference import gamma
from ai_journey.matmul_metal_runtime import SIDES, TORCH_MATMUL, WorkloadCheck
from ai_journey.memory_access_benchmark import INJECTED_TIMER, TIMER
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_recorded import RecordedBackend
from test_matmul_metal_runtime import matmul_runtime

COLUMNS, ROWS = "matmul_x_columns", "matmul_x_rows"


class ScriptedWorkload:
    """Each call advances a scripted clock by its side's cost and is logged."""

    def __init__(self, costs, check=None):
        self.costs = costs
        self.now = 0
        self.events = []
        self.sides = {name: self._call(name) for name in SIDES}
        self.check = check or (lambda: WorkloadCheck(True, 0.0))

    def _call(self, name):
        def call():
            self.now += self.costs[name]
            self.events.append(name)

        return call

    def synchronize(self):
        self.events.append("sync")

    def clock(self):
        return self.now


def scripted_sweep(backend, shapes, **changes):
    controls = {
        "seed": 74,
        "shapes": tuple(matmul_shape(*shape) for shape in shapes),
        "group_size": (32, 8),
        "warmup": 1,
        "repeats": 3,
        "target_flops_per_sample": 1 << 20,
        "clock": itertools.count(0, 1000).__next__,
    }
    return benchmark_sweep(backend, **{**controls, **changes})


def test_a_sample_holds_the_fewest_calls_that_reach_the_flops_target():
    assert calls_per_sample(matmul_shape(128, 128, 128).flops, 1 << 30) == 256
    assert calls_per_sample(matmul_shape(4096, 4096, 4096).flops, 1 << 30) == 1
    assert calls_per_sample(3 * 2**20, 1 << 20) == 1
    with pytest.raises(MatmulMetalError, match="at most 4096 are allowed"):
        calls_per_sample(2, 1 << 20)
    for target in ((1 << 20) - 1, (1 << 40) + 1):
        with pytest.raises(KernelMetalError, match="target_flops_per_sample"):
            calls_per_sample(1024, target)


def test_samples_synchronize_around_each_batch_and_rotate_the_sides():
    workload = ScriptedWorkload({COLUMNS: 100, ROWS: 300, TORCH_MATMUL: 10})

    timings = time_sides(
        workload, calls_per_sample=2, warmup=1, repeats=3, clock=workload.clock
    )

    runs = [(name, len(list(run))) for name, run in itertools.groupby(workload.events)]
    batches = [(name, calls) for name, calls in runs if name != "sync"]
    order = [name for name, _ in batches]
    assert order == [
        *SIDES,
        COLUMNS,
        ROWS,
        TORCH_MATMUL,
        ROWS,
        TORCH_MATMUL,
        COLUMNS,
        TORCH_MATMUL,
        COLUMNS,
        ROWS,
    ]
    assert {calls for _, calls in batches} == {2}
    assert workload.events.count("sync") == 2 * 12
    assert [timing.implementation for timing in timings] == list(SIDES)
    assert [timing.samples_ns for timing in timings] == [
        (200, 200, 200),
        (600, 600, 600),
        (20, 20, 20),
    ]
    assert {timing.timer for timing in timings} == {INJECTED_TIMER}
    assert {timing.calls_per_sample for timing in timings} == {2}


def test_timing_refuses_a_stalled_clock_a_spent_budget_and_bad_sides():
    workload = ScriptedWorkload({COLUMNS: 0, ROWS: 1, TORCH_MATMUL: 1})
    with pytest.raises(MatmulMetalError, match="did not advance"):
        time_sides(
            workload, calls_per_sample=1, warmup=0, repeats=3, clock=workload.clock
        )
    slow = ScriptedWorkload({COLUMNS: 10**9, ROWS: 10**9, TORCH_MATMUL: 10**9})
    budget = RunBudget(1, slow.clock)
    with pytest.raises(KernelMetalError, match="max_runtime_seconds=1"):
        time_sides(
            slow,
            calls_per_sample=1,
            warmup=1,
            repeats=3,
            clock=slow.clock,
            budget=budget,
        )
    swapped = ScriptedWorkload({COLUMNS: 1, ROWS: 1, TORCH_MATMUL: 1})
    swapped.sides = dict(reversed(list(swapped.sides.items())))
    with pytest.raises(MatmulMetalError, match="in order"):
        time_sides(
            swapped, calls_per_sample=1, warmup=0, repeats=3, clock=swapped.clock
        )
    for bad in ({"repeats": 2}, {"warmup": 11}, {"calls_per_sample": 0}):
        controls = {"calls_per_sample": 1, "warmup": 0, "repeats": 3, **bad}
        with pytest.raises(KernelMetalError, match="must be in"):
            time_sides(swapped, clock=swapped.clock, **controls)


def test_a_point_reports_gflops_iqr_and_ratios_from_its_samples():
    shape = matmul_shape(64, 32, 16)
    workload = ScriptedWorkload({COLUMNS: 1_000, ROWS: 4_000, TORCH_MATMUL: 250})
    timings = time_sides(
        workload, calls_per_sample=4, warmup=0, repeats=3, clock=workload.clock
    )
    point = MatmulPoint(0, shape, (), timings, 1e-7)

    mapping = point.to_mapping()

    flops = 2 * 64 * 32 * 16
    assert mapping["flops_per_call"] == flops
    columns = mapping["sides"][COLUMNS]
    assert columns["flops_per_sample"] == 4 * flops
    assert columns["median_ns_per_call"] == 1_000.0
    assert columns["median_gflops_per_second"] == 4 * flops / 4_000
    assert columns["iqr_gflops_per_second"] == [4 * flops / 4_000] * 2
    assert mapping["naive_to_torch_gflops_ratio"] == {COLUMNS: 0.25, ROWS: 0.0625}
    assert mapping["x_rows_to_x_columns_gflops_ratio"] == 0.25
    assert mapping["torch_disagreement"] == 1e-7
    assert mapping["agreement_bound"] == agreement_bound(16) == 4 * gamma(16, 2**-24)
    assert mapping["access_steps"]["x_rows"]["a_step_bytes"] == {"x": 64, "y": 0}


def test_a_sweep_times_each_shape_on_its_own_operands():
    backend = RecordedBackend()

    points = scripted_sweep(backend, [(33, 17, 9), (16, 64, 8)])

    assert [point.shape.to_mapping() for point in points] == [
        {"m": 33, "n": 17, "k": 9},
        {"m": 16, "n": 64, "k": 8},
    ]
    mapping = points[0].to_mapping()
    assert mapping["launches"]["x_columns"]["extent"] == [17, 33]
    assert mapping["launches"]["x_rows"]["extent"] == [33, 17]
    # ceil(2**20 / (2 * 33 * 17 * 9)) calls reach the FLOPs target.
    assert mapping["sides"][COLUMNS]["calls_per_sample"] == 104
    assert {name for name, _ in backend.launches} == {COLUMNS, ROWS}
    assert {point.torch_disagreement for point in points} == {0.0}


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (WorkloadCheck(False, 0.0), MatmulMetalError, "naive outputs differ"),
        (WorkloadCheck(True, 1.0), MatmulMetalError, "agreement bound"),
        (WorkloadCheck(True, float("nan")), MatmulMetalError, "agreement bound"),
        (True, TypeError, "WorkloadCheck"),
    ],
)
def test_a_point_whose_timed_outputs_disagree_stops_the_sweep(result, error, message):
    class Disagrees(RecordedBackend):
        def workload(self, left, right, launches):
            bound = super().workload(left, right, launches)
            return type(bound)(bound.sides, bound.synchronize, lambda: result)

    with pytest.raises(error, match=message):
        scripted_sweep(Disagrees(), [(8, 8, 8)])


def test_the_ceiling_is_the_best_torch_point_and_fractions_follow():
    shape = matmul_shape(64, 64, 64)
    points = []
    for index, costs in enumerate(
        [
            {COLUMNS: 400, ROWS: 800, TORCH_MATMUL: 100},
            {COLUMNS: 200, ROWS: 1000, TORCH_MATMUL: 50},
            {COLUMNS: 300, ROWS: 600, TORCH_MATMUL: 50},
        ]
    ):
        workload = ScriptedWorkload(costs)
        timings = time_sides(
            workload, calls_per_sample=1, warmup=0, repeats=3, clock=workload.clock
        )
        points.append(MatmulPoint(index, shape, (), timings, 0.0))

    summary = ceiling_summary(tuple(points))

    flops = shape.flops
    assert summary["measured_ceiling_gflops_per_second"] == flops / 50
    assert summary["ceiling_index"] == 1
    assert summary["naive"][COLUMNS] == {
        "best_median_gflops_per_second": flops / 200,
        "best_index": 1,
        "fraction_of_measured_ceiling": (flops / 200) / (flops / 50),
    }
    assert summary["naive"][ROWS]["best_index"] == 2
    assert "not a hardware or vendor peak" in summary["definition"]
    with pytest.raises(MatmulMetalError, match="at least one sweep point"):
        ceiling_summary(())


def test_the_ceiling_follows_torch_matmul_where_a_naive_kernel_peaks_elsewhere():
    # x_columns peaks at point 0 and torch.matmul at point 1, so only the
    # torch point can give the ceiling.
    shape = matmul_shape(64, 64, 64)
    points = []
    for index, costs in enumerate(
        [
            {COLUMNS: 100, ROWS: 300, TORCH_MATMUL: 80},
            {COLUMNS: 400, ROWS: 200, TORCH_MATMUL: 20},
        ]
    ):
        workload = ScriptedWorkload(costs)
        timings = time_sides(
            workload, calls_per_sample=1, warmup=0, repeats=3, clock=workload.clock
        )
        points.append(MatmulPoint(index, shape, (), timings, 0.0))

    summary = ceiling_summary(tuple(points))

    flops = shape.flops
    assert (
        summary["ceiling_index"],
        summary["measured_ceiling_gflops_per_second"],
    ) == (
        1,
        flops / 20,
    )
    assert summary["naive"][COLUMNS] == {
        "best_median_gflops_per_second": flops / 100,
        "best_index": 0,
        "fraction_of_measured_ceiling": (flops / 100) / (flops / 20),
    }
    assert summary["naive"][ROWS]["best_index"] == 1


@requires_metal
def test_this_gpu_times_a_small_sweep_with_the_host_timer():
    points = scripted_sweep(
        matmul_runtime(), [(64, 64, 64), (33, 17, 65)], clock=time.perf_counter_ns
    )

    for point in points:
        assert {timing.timer for timing in point.timings} == {TIMER}
        assert 0.0 <= point.torch_disagreement <= agreement_bound(point.shape.k)
        for side in SIDES:
            assert point.median_gflops(side) > 0
