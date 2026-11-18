"""Time the empty kernel, then every Day 74 matmul side at every sweep shape.

The empty kernel comes first: one one-thread launch that does no work, timed
on all three clocks. It gives the floor of one ``compile_shader`` launch on
this host, with no matmul in it. Each sweep shape then binds the Day 74
workload on fresh operands from the Day 74 benchmark stream: both naive
mappings and ``torch.matmul(a, b, out=c)``, each timed on all three clocks.

After the timing of a shape, the latest outputs must pass the Day 74 check:
the two naive kernels hold the same bits, and ``torch.matmul`` stays within
the Day 74 agreement bound. A timing of wrong results is not a timing, so a
shape that fails stops the run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .gpu_timing_benchmark import SideTiming, time_sides
from .gpu_timing_clocks import GpuTimingError
from .gpu_timing_source import EMPTY_KERNEL, empty_kernel_mapping, empty_launch
from .kernel_metal_benchmark import MAX_POINTS, RunBudget
from .kernel_metal_geometry import CheckedLaunch
from .matmul_metal_benchmark import _check, agreement_bound
from .matmul_metal_mapping import MAPPINGS, matmul_launch, matmul_shape
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_runtime import SIDES
from .matmul_metal_source import KERNEL_FOR_MAPPING
from .memory_access_tiling import MatmulShape


@dataclass(frozen=True)
class EmptyPoint:
    """The empty kernel's one-thread launch and its three-clock timing."""

    launch: CheckedLaunch
    timing: SideTiming

    def __post_init__(self) -> None:
        if self.timing.implementation != EMPTY_KERNEL:
            raise GpuTimingError("the empty point must time the empty kernel")

    def to_mapping(self) -> dict[str, object]:
        """Return the kernel, its checked launch and its timing."""

        return {
            "kernel": empty_kernel_mapping(),
            "launch": self.launch.to_mapping(),
            "timing": self.timing.to_mapping(),
        }


@dataclass(frozen=True)
class SweepPoint:
    """One shape: three sides timed on the same device operands."""

    index: int
    shape: MatmulShape
    launches: tuple[CheckedLaunch, ...]
    timings: tuple[SideTiming, ...]
    torch_disagreement: float

    def __post_init__(self) -> None:
        if tuple(item.implementation for item in self.timings) != SIDES:
            raise GpuTimingError(f"a sweep point must time {', '.join(SIDES)}")

    def timing(self, side: str) -> SideTiming:
        """Return one side's timing."""

        return self.timings[SIDES.index(side)]

    def to_mapping(self) -> dict[str, object]:
        """Return the shape, launches, raw samples and the agreement check."""

        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "flops_per_call": self.shape.flops,
            "launches": {
                mapping: launch.to_mapping()
                for mapping, launch in zip(MAPPINGS, self.launches)
            },
            "sides": {item.implementation: item.to_mapping() for item in self.timings},
            "torch_disagreement": self.torch_disagreement,
            "agreement_bound": agreement_bound(self.shape.k),
        }


def measure_sweep(
    backend: object,
    *,
    seed: int,
    shapes: tuple[MatmulShape, ...],
    group_size: tuple[int, int],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[EmptyPoint, tuple[SweepPoint, ...]]:
    """Time the empty kernel and then every shape, in sweep order."""

    if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_POINTS:
        raise GpuTimingError(f"a sweep holds 1 to {MAX_POINTS} shapes")
    common = {
        "synchronize": backend.synchronize,
        "event_pair": backend.event_pair,
        "warmup": warmup,
        "repeats": repeats,
        "clock": clock,
        "budget": budget,
    }
    launch = CheckedLaunch(empty_launch(), backend.limits[EMPTY_KERNEL])
    (empty,) = time_sides({EMPTY_KERNEL: backend.empty_side(launch)}, **common)
    points = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, MatmulShape):
            raise TypeError("a sweep shape must be a MatmulShape")
        shape = matmul_shape(shape.m, shape.n, shape.k)
        left, right = matmul_inputs(seed, BENCHMARK_STREAM, index, shape)
        launches = tuple(
            CheckedLaunch(
                matmul_launch(mapping, shape, group_size),
                backend.limits[KERNEL_FOR_MAPPING[mapping]],
            )
            for mapping in MAPPINGS
        )
        workload = backend.workload(
            left,
            right,
            {
                KERNEL_FOR_MAPPING[mapping]: item
                for mapping, item in zip(MAPPINGS, launches)
            },
        )
        timings = time_sides(
            dict(workload.sides), **{**common, "synchronize": workload.synchronize}
        )
        label = f"point {index} ({shape.m} x {shape.n} x {shape.k})"
        disagreement = _check(workload.check(), label, shape.k)
        points.append(SweepPoint(index, shape, launches, timings, disagreement))
        del workload
    return EmptyPoint(launch, empty), tuple(points)
