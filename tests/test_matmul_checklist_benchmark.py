import itertools
import time

import pytest

from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_checklist_benchmark import (
    ChecklistPoint,
    benchmark_point,
    benchmark_sweep,
    modeled_flops_per_byte,
    modeled_loads_per_output,
)
from ai_journey.matmul_checklist_runtime import ChecklistWorkload
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_metal_benchmark import agreement_bound
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_benchmark import global_loads_per_output
from ai_journey.matmul_tiled_runtime import SideCheck
from ai_journey.matmul_tiled_source import MatmulTiledError
from test_kernel_metal_runtime import requires_metal
from test_matmul_checklist_correctness import StandInBackend
from test_matmul_checklist_runtime import VECTOR, checklist_runtime

KERNELS = tuple(VARIANTS)
SIDES = [*KERNELS, TORCH_MATMUL]


class LoggingBackend(StandInBackend):
    """Log every side call and every synchronize, in the order they happen."""

    def __init__(self, *args, check=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []
        self._check = check

    def synchronize(self):
        self.log.append("sync")

    def workload(self, left, right, launches):
        inner = super().workload(left, right, launches)

        def call(name):
            return lambda: self.log.append(name)

        return ChecklistWorkload(
            sides={name: call(name) for name in inner.sides},
            synchronize=self.synchronize,
            check=self._check or inner.check,
        )


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def point(backend, shape=matmul_shape(40, 24, 72), names=KERNELS, **changes):
    controls = {
        "seed": 77,
        "index": 0,
        "shape": shape,
        "naive_group_size": (32, 8),
        "names": names,
        "warmup": 1,
        "repeats": 3,
        "target_flops_per_sample": 1 << 20,
        "clock": scripted_clock(),
    }
    return benchmark_point(backend, **{**controls, **changes})


def test_each_side_warms_up_then_takes_turns_shifted_by_one_per_repeat():
    backend = LoggingBackend()
    shape = matmul_shape(40, 24, 72)

    result = point(backend, shape)

    calls = -(-(1 << 20) // shape.flops)
    batches = [entry for entry in backend.log if entry != "sync"]
    order = batches[::calls]
    assert calls == 8 and len(batches) == calls * len(SIDES) * (1 + 3)
    assert order[: len(SIDES)] == SIDES
    timed = order[len(SIDES) :]
    for repeat in range(3):
        turn = timed[repeat * len(SIDES) : (repeat + 1) * len(SIDES)]
        assert turn == SIDES[repeat:] + SIDES[:repeat]
    assert result.sides == tuple(SIDES)
    assert all(timing.calls_per_sample == calls for timing in result.timings)
    assert all(len(timing.samples_ns) == 3 for timing in result.timings)


def test_a_point_records_launches_model_rates_and_checks():
    shape = matmul_shape(65, 68, 40)
    mapping = point(StandInBackend(), shape).to_mapping()

    assert mapping["side_order"] == SIDES
    assert list(mapping["launches"]) == list(KERNELS)
    assert list(mapping["model"]) == list(KERNELS)
    assert list(mapping["sides"]) == SIDES
    assert list(mapping["checks"]) == SIDES[1:]
    assert mapping["agreement_bound"] == agreement_bound(40)
    assert mapping["launches"]["matmul_s2_coarse_4x1"]["group_size"] == [32, 8]
    assert mapping["model"]["matmul_x_rows"] == {
        "loads_per_output": 80.0,
        "flops_per_byte": 80 / (4 * 81),
    }
    torch = mapping["sides"][TORCH_MATMUL]
    assert (
        torch["median_gflops_per_second"]
        == torch["flops_per_sample"] / (torch["timing"]["median_ns"])
    )


def test_the_model_counts_device_loads_without_a_cache():
    cube = matmul_shape(4096, 4096, 4096)
    ragged = matmul_shape(2000, 3000, 1000)

    for name in ("matmul_x_columns", "matmul_x_rows"):
        assert modeled_loads_per_output(VARIANTS[name], cube) == 8192.0
    assert modeled_loads_per_output(VARIANTS["matmul_tiled_32"], ragged) == (
        global_loads_per_output(ragged, 32)
    )
    loads = {
        name: modeled_loads_per_output(VARIANTS[name], cube)
        for name in (
            "matmul_s2_coarse_4x1",
            "matmul_s3_block_4x4",
            "matmul_s6_group_8x8",
            "matmul_s6_group_32x32",
            "matmul_s7_interior",
        )
    }
    assert loads == {
        "matmul_s2_coarse_4x1": 256.0,
        "matmul_s3_block_4x4": 128.0,
        "matmul_s6_group_8x8": 256.0,
        "matmul_s6_group_32x32": 64.0,
        "matmul_s7_interior": 128.0,
    }
    assert round(modeled_flops_per_byte(VARIANTS["matmul_s3_block_4x4"], cube), 1) == (
        15.9
    )
    assert modeled_flops_per_byte(VARIANTS["matmul_x_columns"], cube) < 0.25


def test_a_side_that_breaks_the_agreement_bound_stops_the_point():
    def far():
        results = {name: SideCheck(True, 0.0) for name in SIDES[1:]}
        results["matmul_s5_pad4"] = SideCheck(False, 1.0)
        return results

    def claims():
        results = {name: SideCheck(True, 0.0) for name in SIDES[1:]}
        results["matmul_s3_block_4x4"] = SideCheck(True, 2.0**-30)
        return results

    with pytest.raises(MatmulTiledError, match="agreement bound"):
        point(LoggingBackend(check=far))
    with pytest.raises(MatmulTiledError, match="cannot hold the naive bits"):
        point(LoggingBackend(check=claims))


def test_only_admitted_kernels_after_s0_are_timed():
    backend = StandInBackend(threads={"matmul_s6_group_32x32": 512})
    admitted = tuple(name for name in KERNELS if name != "matmul_s6_group_32x32")

    result = point(backend, names=admitted)

    assert "matmul_s6_group_32x32" not in result.sides
    with pytest.raises(MatmulChecklistError, match="refused by its pipeline"):
        point(backend)
    with pytest.raises(MatmulChecklistError, match="times matmul_x_columns first"):
        point(StandInBackend(), names=KERNELS[1:])


def test_a_sweep_stops_between_samples_when_its_budget_runs_out():
    controls = {
        "seed": 77,
        "shapes": (matmul_shape(40, 24, 72), matmul_shape(65, 68, 40)),
        "naive_group_size": (32, 8),
        "names": KERNELS,
        "warmup": 1,
        "repeats": 3,
        "target_flops_per_sample": 1 << 20,
    }
    points = benchmark_sweep(StandInBackend(), clock=scripted_clock(), **controls)
    assert [entry.index for entry in points] == [0, 1]
    assert all(isinstance(entry, ChecklistPoint) for entry in points)

    clock = scripted_clock(step=10**8)
    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        benchmark_sweep(
            StandInBackend(), clock=clock, budget=RunBudget(1, clock), **controls
        )
    with pytest.raises(MatmulChecklistError, match="1 to 16 shapes"):
        benchmark_sweep(StandInBackend(), **{**controls, "shapes": ()})


@requires_metal
def test_a_real_point_times_every_side_and_agrees_with_s0():
    runtime = checklist_runtime()
    names = tuple(
        name
        for name in KERNELS
        if name not in runtime.admissions or runtime.admissions[name].admitted
    )

    result = point(runtime, VECTOR, names=names, clock=time.perf_counter_ns)

    assert result.sides == (*names, TORCH_MATMUL)
    for name, check in result.checks.items():
        assert 0.0 <= check.disagreement <= agreement_bound(VECTOR.k), name
        if name != TORCH_MATMUL:
            assert check.same_bits_as_naive is True, name
