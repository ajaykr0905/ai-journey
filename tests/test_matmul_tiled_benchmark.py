import itertools
import time

import pytest

from ai_journey.kernel_metal_benchmark import RunBudget
from ai_journey.kernel_metal_geometry import KernelMetalError
from ai_journey.matmul_metal_benchmark import agreement_bound
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_runtime import TORCH_MATMUL
from ai_journey.matmul_tiled_benchmark import (
    benchmark_point,
    benchmark_sweep,
    global_loads_per_output,
)
from ai_journey.matmul_tiled_runtime import NAIVE_KERNEL, SideCheck, TiledWorkload
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_variants import VARIANTS
from test_kernel_metal_runtime import requires_metal
from test_matmul_tiled_correctness import ALL_VARIANTS, StandInBackend
from test_matmul_tiled_runtime import tiled_runtime

SIDES = [NAIVE_KERNEL, *VARIANTS, TORCH_MATMUL]


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

        return TiledWorkload(
            sides={name: call(name) for name in inner.sides},
            synchronize=self.synchronize,
            check=self._check or inner.check,
        )


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def point(backend, shape=matmul_shape(40, 24, 72), **changes):
    controls = {
        "seed": 76,
        "index": 0,
        "shape": shape,
        "naive_group_size": (32, 8),
        "variants": ALL_VARIANTS,
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
    assert all(timing.samples_ns == (1_000,) * 3 for timing in result.timings)
    assert {timing.timer for timing in result.timings} == {"injected"}


def test_a_point_records_its_rates_checks_and_modeled_loads():
    shape = matmul_shape(40, 24, 72)

    mapping = point(StandInBackend(), shape).to_mapping()

    assert list(mapping["sides"]) == mapping["side_order"] == SIDES
    assert list(mapping["launches"]) == SIDES[:-1]
    assert mapping["launches"]["matmul_tiled_32"]["group_size"] == [32, 32]
    entry = mapping["sides"]["matmul_tiled_16"]
    assert entry["calls_per_sample"] == 8 and entry["flops_per_sample"] == 8 * 138240
    assert entry["median_gflops_per_second"] == 8 * 138240 / 1_000
    assert mapping["checks"] == {
        "matmul_tiled_8": {"same_bits_as_naive": True, "disagreement": 0.0},
        "matmul_tiled_16": {"same_bits_as_naive": True, "disagreement": 0.0},
        "matmul_tiled_32": {"same_bits_as_naive": True, "disagreement": 0.0},
        TORCH_MATMUL: {"same_bits_as_naive": False, "disagreement": 2.0**-30},
    }
    assert mapping["agreement_bound"] == agreement_bound(72)
    assert mapping["global_loads_per_output"] == {
        NAIVE_KERNEL: 144.0,
        "matmul_tiled_8": (3 * 40 * 72 + 5 * 72 * 24) / (40 * 24),
        "matmul_tiled_16": (2 * 40 * 72 + 3 * 72 * 24) / (40 * 24),
        "matmul_tiled_32": (1 * 40 * 72 + 2 * 72 * 24) / (40 * 24),
    }


def test_the_model_divides_global_loads_per_output_by_about_the_tile():
    shape = matmul_shape(4096, 4096, 4096)

    assert global_loads_per_output(shape, None) == 8192.0
    for tile in (8, 16, 32):
        assert global_loads_per_output(shape, tile) == 8192 / tile


def test_a_side_past_the_agreement_bound_or_a_partial_check_stops_the_run():
    far = {name: SideCheck(True, 0.0) for name in (*VARIANTS, TORCH_MATMUL)} | {
        TORCH_MATMUL: SideCheck(False, 1e-3)
    }
    with pytest.raises(MatmulTiledError, match="torch_matmul sits 0.001"):
        point(LoggingBackend(check=lambda: far))
    partial = {name: SideCheck(True, 0.0) for name in VARIANTS}
    with pytest.raises(MatmulTiledError, match="cover every side"):
        point(LoggingBackend(check=lambda: partial))
    nan = {name: SideCheck(True, float("nan")) for name in (*VARIANTS, TORCH_MATMUL)}
    with pytest.raises(MatmulTiledError, match="sits nan"):
        point(LoggingBackend(check=lambda: nan))


def test_a_check_with_the_naive_bits_and_a_nonzero_disagreement_stops_the_run():
    # Equal bits subtract to 0, so the device check cannot give both values.
    claimed = {name: SideCheck(True, 0.0) for name in (*VARIANTS, TORCH_MATMUL)} | {
        TORCH_MATMUL: SideCheck(True, 2.0**-30)
    }
    signed_zero = {name: SideCheck(False, 0.0) for name in (*VARIANTS, TORCH_MATMUL)}

    with pytest.raises(MatmulTiledError, match="torch_matmul cannot hold the naive"):
        point(LoggingBackend(check=lambda: claimed))
    # Other bits with no difference stay valid: -0.0 and +0.0 differ only in bits.
    assert point(LoggingBackend(check=lambda: signed_zero)).checks == signed_zero


def test_a_sweep_times_only_admitted_variants_within_its_budget():
    backend = StandInBackend(threads={"matmul_tiled_32": 512})
    admitted = ALL_VARIANTS[:2]
    controls = {
        "seed": 76,
        "shapes": (matmul_shape(40, 24, 72), matmul_shape(17, 33, 9)),
        "naive_group_size": (32, 8),
        "warmup": 0,
        "repeats": 3,
        "target_flops_per_sample": 1 << 20,
        "clock": scripted_clock(),
    }

    points = benchmark_sweep(backend, variants=admitted, **controls)

    assert [result.sides for result in points] == [
        (NAIVE_KERNEL, "matmul_tiled_8", "matmul_tiled_16", TORCH_MATMUL)
    ] * 2
    with pytest.raises(MatmulTiledError, match="refused by its pipeline"):
        benchmark_sweep(backend, variants=ALL_VARIANTS, **controls)
    clock = scripted_clock(step=10**9)
    budget = RunBudget(1, clock)
    with pytest.raises(KernelMetalError, match="exceeded max_runtime_seconds=1"):
        benchmark_sweep(
            backend,
            variants=admitted,
            **{**controls, "clock": clock},
            budget=budget,
        )


@requires_metal
def test_one_real_point_times_every_side_and_checks_it_on_the_device():
    runtime = tiled_runtime()
    shape = matmul_shape(64, 48, 100)
    admitted = tuple(
        variant for variant in ALL_VARIANTS if runtime.admissions[variant.name].admitted
    )

    result = point(runtime, shape, variants=admitted, clock=time.perf_counter_ns)

    mapping = result.to_mapping()
    assert result.sides == (
        NAIVE_KERNEL,
        *(item.name for item in admitted),
        TORCH_MATMUL,
    )
    assert {timing.timer for timing in result.timings} == {"time.perf_counter_ns"}
    for name, check in mapping["checks"].items():
        assert 0.0 <= check["disagreement"] <= agreement_bound(100), name
