import itertools
import time

import numpy
import pytest

from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.kernel_metal_reference import (
    array_sha256,
    vector_add_inputs,
    vector_add_reference,
)
from ai_journey.metal_launch_sweep_benchmark import (
    MAX_TARGET_FLOPS,
    MIN_TARGET_FLOPS,
    matmul_calls_per_sample,
    matmul_flops,
    measure_matmul_point,
    measure_vector_point,
    time_shapes,
)
from ai_journey.metal_launch_sweep_correctness import matmul_inputs
from ai_journey.metal_launch_sweep_runtime import SweepWorkload
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError, admit_shapes
from test_kernel_metal_runtime import requires_metal
from test_metal_launch_sweep_correctness import matmul_admissions, vector_admissions
from test_metal_launch_sweep_standin import StandInBackend, rounded_product


class FakeClock:
    """A clock that only the workload's calls advance, by a cost per shape."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


def logged_workload(costs, clock, log):
    def call(position):
        def run():
            log.append(position)
            clock.now += costs[position]

        return run

    return SweepWorkload(
        calls=tuple(call(position) for position in range(len(costs))),
        synchronize=lambda: log.append("sync"),
        outputs_agree=lambda: True,
        output=lambda: None,
    )


def test_batches_reach_the_flop_target_with_whole_calls():
    assert matmul_flops(256, 256, 256) == 2 * 256**3
    assert matmul_calls_per_sample(2 * 256**3, 1 << 30) == 32
    assert matmul_calls_per_sample(2 * 2048**3, 1 << 30) == 1
    assert matmul_calls_per_sample(1000, MIN_TARGET_FLOPS) == 1049
    with pytest.raises(MetalLaunchSweepError, match="524288 calls; at most 4096"):
        matmul_calls_per_sample(2, 1 << 20)
    with pytest.raises(KernelMetalError, match="target_flops_per_sample"):
        matmul_calls_per_sample(1 << 30, MAX_TARGET_FLOPS + 1)


def test_shapes_take_turns_that_rotate_every_repeat():
    clock, log = FakeClock(), []
    workload = logged_workload((10, 20, 30), clock, log)

    samples = time_shapes(
        workload, calls_per_sample=2, warmup=1, repeats=3, clock=clock
    )

    assert samples == ((20, 20, 20), (40, 40, 40), (60, 60, 60))
    batches = [entry for entry in log if entry != "sync"]
    # One untimed sample per shape, then the turns 0 1 2, 1 2 0 and 2 0 1.
    turns = [0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1]
    assert batches == [position for position in turns for _ in range(2)]
    # Every sample, untimed or timed, waits before and after its batch.
    assert log.count("sync") == 2 * len(turns)
    assert log[:4] == ["sync", 0, 0, "sync"]


def test_a_clock_that_stands_still_or_runs_out_of_budget_stops_the_timing():
    clock, log = FakeClock(), []

    with pytest.raises(MetalLaunchSweepError, match="did not advance"):
        time_shapes(
            logged_workload((0, 5), clock, log),
            calls_per_sample=1,
            warmup=0,
            repeats=3,
            clock=clock,
        )
    budget_clock = itertools.count(0, 10**9).__next__
    with pytest.raises(KernelMetalError, match="max_runtime_seconds=2"):
        time_shapes(
            logged_workload((5,), clock, log),
            calls_per_sample=1,
            warmup=0,
            repeats=3,
            clock=clock,
            budget=RunBudget(2, budget_clock),
        )
    empty = SweepWorkload((), lambda: None, lambda: True, lambda: None)
    with pytest.raises(MetalLaunchSweepError, match="one call per shape"):
        time_shapes(empty, calls_per_sample=1, warmup=0, repeats=3, clock=clock)
    with pytest.raises(TypeError, match="budget must be a RunBudget"):
        time_shapes(
            logged_workload((5,), clock, log),
            calls_per_sample=1,
            warmup=0,
            repeats=3,
            budget=object(),
        )


class CostedBackend(StandInBackend):
    """A stand-in whose matmul calls advance a fake clock by a cost per shape."""

    def __init__(self, costs):
        super().__init__()
        self.clock = FakeClock()
        self.costs = costs

    def matmul_workload(self, a, b, launches):
        base = super().matmul_workload(a, b, launches)

        def call(launch):
            cost = self.costs[launch.geometry.group_size]

            def run():
                self.clock.now += cost

            return run

        return SweepWorkload(
            calls=tuple(call(launch) for launch in launches),
            synchronize=base.synchronize,
            outputs_agree=base.outputs_agree,
            output=base.output,
        )


def test_a_matmul_point_ranks_its_shapes_and_keeps_every_rate():
    shapes = ((16, 16), (64, 4), (4, 64))
    backend = CostedBackend({(16, 16): 300, (64, 4): 200, (4, 64): 400})

    point = measure_matmul_point(
        backend,
        seed=74,
        index=0,
        shape=(64, 64, 64),
        admissions=matmul_admissions(backend, shapes),
        warmup=1,
        repeats=3,
        target_flops_per_sample=1 << 20,
        clock=backend.clock,
    ).to_mapping()

    flops = 2 * 64**3
    calls = -(-(1 << 20) // flops)
    assert (point["kernel"], point["m"], point["n"], point["k"]) == (
        "matmul_naive",
        64,
        64,
        64,
    )
    assert (point["work_per_call"], point["calls_per_sample"]) == (flops, calls)
    assert point["ranking"] == [[64, 4], [16, 16], [4, 64]]
    assert (point["spread"], point["separated"]) == (2.0, True)
    first = point["shapes"][0]
    assert first["group_size"] == [16, 16] and first["rank"] == 2
    assert first["samples_ns"] == [300 * calls] * 3
    assert first["median_ns_per_call"] == 300.0
    assert first["median_gflops_per_second"] == flops * calls / (300 * calls)
    assert first["relative_to_fastest"] == 1.5
    assert first["rows_per_simd_group"] == 2
    a, b = matmul_inputs(74, 32, 64, 64, 64)
    assert point["output_sha256"] == array_sha256(rounded_product(a, b))
    assert point["accuracy"]["output_sha256"] == point["output_sha256"]
    assert 0 < point["accuracy"]["max_bound_ratio"] < 1


def test_shapes_with_overlapping_quartiles_are_not_separated():
    backend = CostedBackend({(16, 16): 300, (64, 4): 300})

    point = measure_matmul_point(
        backend,
        seed=74,
        index=1,
        shape=(32, 32, 32),
        admissions=matmul_admissions(backend, ((16, 16), (64, 4))),
        warmup=0,
        repeats=3,
        target_flops_per_sample=1 << 20,
        clock=backend.clock,
    ).to_mapping()

    # Equal medians keep the config order.
    assert point["ranking"] == [[16, 16], [64, 4]]
    assert (point["spread"], point["separated"]) == (1.0, False)


def test_a_vector_point_checks_every_size_against_the_float32_sum():
    backend = StandInBackend()

    point = measure_vector_point(
        backend,
        seed=74,
        index=0,
        length=4096,
        admissions=vector_admissions(backend, ((8,), (48,), (256,))),
        warmup=0,
        repeats=3,
        target_bytes_per_sample=1 << 16,
        clock=itertools.count(0, 1000).__next__,
    ).to_mapping()

    first, second = vector_add_inputs(74, 32, 4096)
    assert point["output_sha256"] == array_sha256(vector_add_reference(first, second))
    assert (point["work_per_call"], point["calls_per_sample"]) == (12 * 4096, 2)
    assert [shape["group_size"] for shape in point["shapes"]] == [[8], [48], [256]]
    assert point["shapes"][1]["idle_lanes"] == 16
    assert point["shapes"][0]["median_gb_per_second"] == 12 * 4096 * 2 / 1000
    assert "accuracy" not in point
    assert {launch.geometry.extent for launch in backend.launches} == {(4096,)}


class Disagreeing(StandInBackend):
    def vector_workload(self, first, second, launches):
        base = super().vector_workload(first, second, launches)
        return SweepWorkload(base.calls, base.synchronize, lambda: False, base.output)


class Off(StandInBackend):
    def vector_workload(self, first, second, launches):
        base = super().vector_workload(first, second, launches)
        return SweepWorkload(
            base.calls,
            base.synchronize,
            base.outputs_agree,
            lambda: numpy.zeros_like(first),
        )


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (Disagreeing(), "differ between sizes"),
        (Off(), "differs from the float32 sum"),
    ],
)
def test_a_timed_vector_output_must_agree_and_be_exact(backend, message):
    with pytest.raises(MetalLaunchSweepError, match=message):
        measure_vector_point(
            backend,
            seed=74,
            index=0,
            length=1000,
            admissions=vector_admissions(backend, ((32,), (64,))),
            warmup=0,
            repeats=3,
            target_bytes_per_sample=1 << 10,
            clock=itertools.count(0, 1000).__next__,
        )


@requires_metal
def test_a_small_point_on_this_gpu_is_timed_with_the_host_clock():
    from test_metal_launch_sweep_runtime import sweep_runtime

    runtime = sweep_runtime()
    admissions = admit_shapes(
        ((16, 16), (64, 4)),
        runtime.limits["matmul_naive"],
        requires_whole_simd_groups=False,
    )

    point = measure_matmul_point(
        runtime,
        seed=74,
        index=0,
        shape=(64, 64, 64),
        admissions=admissions,
        warmup=0,
        repeats=3,
        target_flops_per_sample=1 << 20,
        clock=time.perf_counter_ns,
    )
    vector = measure_vector_point(
        runtime,
        seed=74,
        index=0,
        length=4096,
        admissions=vector_admissions(runtime, ((32,), (256,))),
        warmup=0,
        repeats=3,
        target_bytes_per_sample=1 << 16,
        clock=time.perf_counter_ns,
    )

    assert point.timer == vector.timer == "time.perf_counter_ns"
    for timing in (*point.timings, *vector.timings):
        assert len(timing.samples_ns) == 3 and min(timing.samples_ns) > 0
