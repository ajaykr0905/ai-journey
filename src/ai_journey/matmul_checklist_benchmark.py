"""Time every checklist kernel and ``torch.matmul`` in one rotated run.

Each point draws one pair of seeded operands, places it on the device once
and binds every side to it: the S0 kernel ``matmul_x_columns`` first, then
every other admitted kernel in registry order, then
``torch.matmul(a, b, out=c)`` into a preallocated output. The timing is the
Day 74 and Day 76 method, run by the Day 74 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again.
3. Repeat r starts with side r mod S and runs each of the S sides once in
   order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose FLOPs reach
``target_flops_per_sample``. After the last sample, every side's latest
output is compared on the device with the S0 output, with the Day 76 check:
it must stay within ``4 gamma_K(2**-24)`` of it, relative to ``|A| @ |B|``,
or the run stops, and equal bits must sit exactly 0 from it.

Each point also records two numbers of a model with no cache for every
kernel: the global element loads for each output and the FLOPs for each
byte of that traffic. A model value is not a measurement. Only the host
timer ``time.perf_counter_ns`` makes a measurement; any other clock is
named ``injected``. These are host-specific Apple GPU (Metal) timings;
nothing here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .kernel_metal_benchmark import MAX_POINTS, DeviceTiming, RunBudget
from .kernel_metal_geometry import CheckedLaunch, ceil_div
from .matmul_checklist_correctness import runnable
from .matmul_checklist_runtime import checked_launch
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_steps import REFERENCE
from .matmul_checklist_variants import VARIANTS, ChecklistVariant
from .matmul_metal_benchmark import _rates, agreement_bound, calls_per_sample
from .matmul_metal_mapping import FLOAT32_BYTES, matmul_shape
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_tiled_benchmark import _checks
from .matmul_tiled_runtime import SideCheck
from .memory_access_benchmark import gflops_per_second, timer_name
from .memory_access_tiling import MatmulShape
from .metal_launch_sweep_benchmark import time_shapes

TRAFFIC_MODEL = (
    "analytical global element loads for each output element, with no cache "
    "modeled: 2 * K for a Day 74 kernel; (ceil(N / BN) * M * K + ceil(M / BM) "
    "* K * N) / (M * N) for a kernel with BM x BN tiles; FLOPs for each byte "
    "divide 2 * K by 4 bytes times those loads plus one store"
)


def modeled_loads_per_output(variant: ChecklistVariant, shape: MatmulShape) -> float:
    """Return the no-cache model's global element loads for each output."""

    if variant.design is None and variant.group_size is None:
        return float(2 * shape.k)
    if variant.design is None:
        rows = columns = variant.group_size[0]
    else:
        rows, columns = variant.design.block_m, variant.design.block_n
    useful = (
        ceil_div(shape.n, columns) * shape.m * shape.k
        + ceil_div(shape.m, rows) * shape.k * shape.n
    )
    return useful / (shape.m * shape.n)


def modeled_flops_per_byte(variant: ChecklistVariant, shape: MatmulShape) -> float:
    """Return 2 K FLOPs over the modeled bytes of one output and its store."""

    loads = modeled_loads_per_output(variant, shape)
    return 2 * shape.k / (FLOAT32_BYTES * (loads + 1))


@dataclass(frozen=True)
class ChecklistPoint:
    """One sweep point: every side timed on the same device operands."""

    index: int
    shape: MatmulShape
    launches: Mapping[str, CheckedLaunch]
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: the kernels, then ``torch.matmul``."""

        return tuple(timing.implementation for timing in self.timings)

    def median_gflops(self, side: str) -> float:
        """Return one side's median rate in GFLOP/s."""

        timing = self.timings[self.sides.index(side)]
        flops = self.shape.flops * timing.calls_per_sample
        return gflops_per_second(flops, timing.summary.median_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics, rates, model and checks."""

        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "flops_per_call": self.shape.flops,
            "side_order": list(self.sides),
            "launches": {
                name: launch.to_mapping() for name, launch in self.launches.items()
            },
            "model": {
                name: {
                    "loads_per_output": modeled_loads_per_output(
                        VARIANTS[name], self.shape
                    ),
                    "flops_per_byte": modeled_flops_per_byte(
                        VARIANTS[name], self.shape
                    ),
                }
                for name in self.launches
            },
            "sides": {
                timing.implementation: _rates(timing, self.shape.flops)
                for timing in self.timings
            },
            "checks": {
                name: {
                    "same_bits_as_naive": check.same_bits_as_naive,
                    "disagreement": check.disagreement,
                }
                for name, check in self.checks.items()
            },
            "agreement_bound": agreement_bound(self.shape.k),
        }


def benchmark_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> ChecklistPoint:
    """Time every side of one point on one pair of device operands."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("a sweep shape must be a MatmulShape")
    names = runnable(backend, names)
    if not names or names[0] != REFERENCE:
        raise MatmulChecklistError(f"a point times {REFERENCE} first")
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, BENCHMARK_STREAM, index, shape)
    launches = {
        name: checked_launch(backend.limits, name, shape, naive_group_size)
        for name in names
    }
    workload = backend.workload(left, right, launches)
    sides = tuple(workload.sides)
    if sides != (*names, TORCH_MATMUL):
        raise MatmulChecklistError(
            "a workload must bind every kernel and then torch.matmul"
        )
    calls = calls_per_sample(shape.flops, target_flops_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"point {index} ({shape.m} x {shape.n} x {shape.k})"
    checks = _checks(workload.check(), sides, label, shape.k)
    timer = timer_name(clock)
    timings = tuple(
        DeviceTiming(name, calls, warmup, repeats, series, timer)
        for name, series in zip(sides, samples)
    )
    return ChecklistPoint(index, shape, launches, timings, checks)


def benchmark_sweep(
    backend: object,
    *,
    seed: int,
    shapes: tuple[MatmulShape, ...],
    naive_group_size: tuple[int, int],
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[ChecklistPoint, ...]:
    """Time every side at every sweep shape; ``names`` are the admitted kernels."""

    if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_POINTS:
        raise MatmulChecklistError(f"a sweep holds 1 to {MAX_POINTS} shapes")
    return tuple(
        benchmark_point(
            backend,
            seed=seed,
            index=index,
            shape=shape,
            naive_group_size=naive_group_size,
            names=names,
            warmup=warmup,
            repeats=repeats,
            target_flops_per_sample=target_flops_per_sample,
            clock=clock,
            budget=budget,
        )
        for index, shape in enumerate(shapes)
    )
