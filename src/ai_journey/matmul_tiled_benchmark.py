"""Time the naive kernel, the admitted tiles and ``torch.matmul`` in one rotated run.

Each point draws one pair of seeded operands, places it on the device once
and binds every side to it: the naive Day 74 ``matmul_x_columns``, the faster
Day 74 mapping; each admitted tiled variant in config order; and
``torch.matmul(a, b, out=c)`` into a preallocated output. The timing is the
Day 74 method with more sides, run by the Day 74 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again.
3. Repeat r starts with side r mod S and runs each of the S sides once in
   order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose FLOPs reach
``target_flops_per_sample``, by the Day 74 ``calls_per_sample``. After the
last sample, every side's latest output is compared on the device with the
naive output. It must stay within ``agreement_bound(K) = 4 gamma_K(2**-24)``
of it, relative to ``|A| @ |B|``, or the run stops. Whether a side holds the
naive bits is recorded, not required. Equal bits subtract to 0, so a check
that claims the naive bits with a nonzero disagreement stops the run too.

Each point also records the analytical global element loads per output of
each kernel side, from the Day 74 traffic model: ``2 K`` for the naive kernel
and the useful tile loads over ``M N`` for a tile, about ``2 K / T``. The
model counts no cache. Only the host timer ``time.perf_counter_ns`` makes a
measurement; any other clock is named ``injected``. These are host-specific
Apple GPU (Metal) timings; nothing here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .kernel_metal_benchmark import MAX_POINTS, DeviceTiming, RunBudget
from .kernel_metal_geometry import CheckedLaunch
from .matmul_metal_benchmark import _rates, agreement_bound, calls_per_sample
from .matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_tiled_runtime import NAIVE_KERNEL, SideCheck
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import TiledVariant, require_admitted
from .memory_access_benchmark import gflops_per_second, timer_name
from .memory_access_tiling import MatmulShape, NaiveMatmulTraffic, TiledMatmulTraffic
from .metal_launch_sweep_benchmark import time_shapes

AGREEMENT = (
    "every side within 4 * gamma_K(2**-24) * (|A| @ |B|) of the naive output, "
    "on the device in float32"
)
LOADS_PER_OUTPUT = (
    "analytical global element loads per output element: 2 * K for the naive "
    "kernel; useful tile loads / (M * N) for a T x T tile, from "
    "memory_access_tiling, with no cache modeled"
)


def global_loads_per_output(shape: MatmulShape, tile: int | None) -> float:
    """Return the model's global element loads per output; ``None`` is naive."""

    if tile is None:
        traffic = NaiveMatmulTraffic(shape)
        return traffic.load_elements / traffic.store_elements
    tiled = TiledMatmulTraffic(shape, tile)
    return tiled.load_elements / tiled.store_elements


@dataclass(frozen=True)
class TiledPoint:
    """One sweep point: every side timed on the same device operands."""

    index: int
    shape: MatmulShape
    variants: tuple[TiledVariant, ...]
    launches: tuple[CheckedLaunch, ...]
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: naive, variants, ``torch.matmul``."""

        return tuple(timing.implementation for timing in self.timings)

    def median_gflops(self, side: str) -> float:
        """Return one side's median rate in GFLOP/s."""

        timing = self.timings[self.sides.index(side)]
        flops = self.shape.flops * timing.calls_per_sample
        return gflops_per_second(flops, timing.summary.median_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics, rates and checks."""

        kernels = (NAIVE_KERNEL, *(variant.name for variant in self.variants))
        tiles = (None, *(variant.tile for variant in self.variants))
        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "flops_per_call": self.shape.flops,
            "side_order": list(self.sides),
            "launches": {
                name: launch.to_mapping()
                for name, launch in zip(kernels, self.launches)
            },
            "global_loads_per_output": {
                name: global_loads_per_output(self.shape, tile)
                for name, tile in zip(kernels, tiles)
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


def _checks(
    result: object, sides: tuple[str, ...], label: str, inner: int
) -> dict[str, SideCheck]:
    if not isinstance(result, Mapping) or tuple(result) != sides[1:]:
        raise MatmulTiledError(f"{label}: the check must cover every side in order")
    bound = agreement_bound(inner)
    checks = {}
    for name, check in result.items():
        if (
            not isinstance(check, SideCheck)
            or type(check.same_bits_as_naive) is not bool
        ):
            raise TypeError("a side check must be a SideCheck")
        disagreement = check.disagreement
        if (
            type(disagreement) is not float
            or not math.isfinite(disagreement)
            or not 0.0 <= disagreement <= bound
        ):
            raise MatmulTiledError(
                f"{label}: {name} sits {disagreement!r} from the naive output; "
                f"the agreement bound is {bound!r}"
            )
        # Equal bits subtract to 0, so a check that claims both is broken.
        if check.same_bits_as_naive and disagreement != 0.0:
            raise MatmulTiledError(
                f"{label}: {name} cannot hold the naive bits and sit "
                f"{disagreement!r} from them"
            )
        checks[name] = check
    return checks


def benchmark_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    variants: tuple[TiledVariant, ...],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> TiledPoint:
    """Time every side of one point on one pair of device operands."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("a sweep shape must be a MatmulShape")
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, BENCHMARK_STREAM, index, shape)
    groups = {NAIVE_KERNEL: naive_group_size}
    groups.update({variant.name: variant.group_size for variant in variants})
    launches = {
        name: CheckedLaunch(
            matmul_launch(X_COLUMNS, shape, group), backend.limits[name]
        )
        for name, group in groups.items()
    }
    workload = backend.workload(left, right, launches)
    sides = tuple(workload.sides)
    if sides != (*launches, TORCH_MATMUL):
        raise MatmulTiledError(
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
    return TiledPoint(index, shape, variants, tuple(launches.values()), timings, checks)


def benchmark_sweep(
    backend: object,
    *,
    seed: int,
    shapes: tuple[MatmulShape, ...],
    naive_group_size: tuple[int, int],
    variants: tuple[TiledVariant, ...],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[TiledPoint, ...]:
    """Time every side at every sweep shape; ``variants`` are the admitted ones."""

    if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_POINTS:
        raise MatmulTiledError(f"a sweep holds 1 to {MAX_POINTS} shapes")
    variants = require_admitted(backend.admissions, variants)
    points = []
    for index, shape in enumerate(shapes):
        points.append(
            benchmark_point(
                backend,
                seed=seed,
                index=index,
                shape=shape,
                naive_group_size=naive_group_size,
                variants=variants,
                warmup=warmup,
                repeats=repeats,
                target_flops_per_sample=target_flops_per_sample,
                clock=clock,
                budget=budget,
            )
        )
    return tuple(points)
