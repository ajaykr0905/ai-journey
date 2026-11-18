"""Synchronized GFLOP/s of both naive kernels against ``torch.matmul`` on MPS.

A GPU launch returns once its command is queued, so a host clock read right
after it would time almost nothing. Each timed sample therefore calls
``synchronize`` before its first clock reading, so no earlier work still
runs, launches ``calls_per_sample`` back-to-back calls, and calls
``synchronize`` again before its second reading, so the sample covers their
completion. Each side first runs ``warmup`` untimed samples, which absorb
one-time costs such as building pipelines and MPS graphs. Timed samples then
rotate through the three sides, and the order shifts by one side at every
repeat, so slow drift in clocks or load favors no side. This is the Day 73
timing rule with three sides instead of two, and it reuses that lab's clock
reader, run budget and timing record.

A sample holds the fewest whole calls whose FLOPs reach
``target_flops_per_sample``, at most ``MAX_CALLS_PER_SAMPLE``, so a small
point is not lost in the cost of synchronizing. One call does ``2 * M * N *
K`` FLOPs: one multiply and one add for each term of each output. GFLOP/s is
the FLOPs of a sample divided by its median duration in nanoseconds, and the
IQR of the durations gives a range of rates.

After the last sample, the latest outputs must agree. The two naive kernels
must hold the same bits, and ``torch.matmul`` must stay within
``agreement_bound(K) = 4 gamma_K(2**-24)`` of them, relative to ``|A| @ |B|``.
Each output lies within ``gamma_K(2**-24)`` of the exact product, and the
factor of 2 on top covers the float32 roundings of the check on the device.

The measured ceiling is the highest median GFLOP/s of ``torch.matmul`` over
the sweep: the vendor-optimized path, measured on this host in the same run.
It is not a hardware or vendor peak. Only the host timer
``time.perf_counter_ns`` makes a measurement; any other clock is named
``injected``. These are host-specific Apple GPU (Metal) timings; nothing here
measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat

from .kernel_metal_benchmark import (
    MAX_CALLS_PER_SAMPLE,
    MAX_POINTS,
    MIN_REPEATS,
    DeviceTiming,
    RunBudget,
    _Reader,
)
from .kernel_metal_geometry import CheckedLaunch, require_integer
from .matmul_metal_mapping import (
    MAPPINGS,
    MatmulMetalError,
    access_steps,
    matmul_launch,
    matmul_shape,
)
from .matmul_metal_reference import (
    BENCHMARK_STREAM,
    FLOAT32_UNIT_ROUNDOFF,
    gamma,
    matmul_inputs,
)
from .matmul_metal_runtime import SIDES, TORCH_MATMUL, WorkloadCheck
from .matmul_metal_source import KERNEL_FOR_MAPPING, KERNELS
from .memory_access_benchmark import (
    MAX_REPEATS,
    MAX_WARMUP,
    gflops_per_second,
    timer_name,
)
from .memory_access_tiling import MatmulShape

MIN_TARGET_FLOPS = 1 << 20
MAX_TARGET_FLOPS = 1 << 40
AGREEMENT = "4 * gamma_K(2**-24) * (|A| @ |B|), on the device in float32"
CEILING = (
    "the highest median GFLOP/s of torch.matmul on the MPS device over this "
    "sweep, measured on this host in the same run; not a hardware or vendor peak"
)


def calls_per_sample(flops_per_call: int, target_flops: int) -> int:
    """Return the fewest whole calls whose FLOPs reach the target.

    The batch follows from the shape alone, so a verifier can recompute it.
    """

    per_call = require_integer(flops_per_call, "flops_per_call", 2, 1 << 62)
    target = require_integer(
        target_flops, "target_flops_per_sample", MIN_TARGET_FLOPS, MAX_TARGET_FLOPS
    )
    calls = max(1, -(-target // per_call))
    if calls > MAX_CALLS_PER_SAMPLE:
        raise MatmulMetalError(
            f"a sample would need {calls} calls; at most {MAX_CALLS_PER_SAMPLE} "
            "are allowed"
        )
    return calls


def agreement_bound(inner: int) -> float:
    """Return how far the timed ``torch.matmul`` may sit from the naive output."""

    return 4 * gamma(inner, FLOAT32_UNIT_ROUNDOFF)


def time_sides(
    workload: object,
    *,
    calls_per_sample: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[DeviceTiming, ...]:
    """Time the three sides of one workload in rotating synchronized samples.

    Returns one timing per side in ``SIDES`` order. Every sample, untimed or
    timed, starts and ends with ``workload.synchronize()``.
    """

    calls = require_integer(
        calls_per_sample, "calls_per_sample", 1, MAX_CALLS_PER_SAMPLE
    )
    warm = require_integer(warmup, "warmup", 0, MAX_WARMUP)
    count = require_integer(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    sides = dict(workload.sides)
    if tuple(sides) != SIDES:
        raise MatmulMetalError(f"a workload must bind {', '.join(SIDES)} in order")
    synchronize = workload.synchronize
    read = _Reader(clock, "benchmark")

    def batch(call: Callable[[], object]) -> None:
        for _ in repeat(None, calls):
            call()

    def checkpoint() -> None:
        if budget is not None:
            budget.check()

    for call in sides.values():
        for _ in range(warm):
            synchronize()
            batch(call)
            synchronize()
            checkpoint()
    samples: dict[str, list[int]] = {name: [] for name in SIDES}
    for index in range(count):
        shift = index % len(SIDES)
        for name in SIDES[shift:] + SIDES[:shift]:
            synchronize()
            started = read()
            batch(sides[name])
            synchronize()
            elapsed = read() - started
            if elapsed <= 0:
                raise MatmulMetalError(
                    "benchmark clock did not advance across a sample"
                )
            samples[name].append(elapsed)
            checkpoint()
    timer = timer_name(clock)
    return tuple(
        DeviceTiming(name, calls, warm, count, tuple(samples[name]), timer)
        for name in SIDES
    )


def _rates(timing: DeviceTiming, flops_per_call: int) -> dict[str, object]:
    summary = timing.summary
    calls = timing.calls_per_sample
    flops = flops_per_call * calls
    return {
        **timing.to_mapping(),
        "flops_per_sample": flops,
        "median_ns_per_call": summary.median_ns / calls,
        "iqr_ns_per_call": summary.iqr_ns / calls,
        "median_gflops_per_second": gflops_per_second(flops, summary.median_ns),
        "best_gflops_per_second": gflops_per_second(flops, summary.min_ns),
        "iqr_gflops_per_second": [
            gflops_per_second(flops, summary.p75_ns),
            gflops_per_second(flops, summary.p25_ns),
        ],
    }


@dataclass(frozen=True)
class MatmulPoint:
    """One sweep point: three sides timed on the same device operands."""

    index: int
    shape: MatmulShape
    launches: tuple[CheckedLaunch, ...]
    timings: tuple[DeviceTiming, ...]
    torch_disagreement: float

    def median_ns(self, side: str) -> int:
        """Return one side's median sample duration."""

        return self.timings[SIDES.index(side)].summary.median_ns

    def median_gflops(self, side: str) -> float:
        """Return one side's median rate in GFLOP/s."""

        timing = self.timings[SIDES.index(side)]
        flops = self.shape.flops * timing.calls_per_sample
        return gflops_per_second(flops, timing.summary.median_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics, rates and ratios.

        Every side runs the same calls per sample, so a ratio of rates is the
        inverse ratio of median durations.
        """

        columns, rows = (KERNEL_FOR_MAPPING[mapping] for mapping in MAPPINGS)
        torch_median = self.median_ns(TORCH_MATMUL)
        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "flops_per_call": self.shape.flops,
            "launches": {
                mapping: launch.to_mapping()
                for mapping, launch in zip(MAPPINGS, self.launches)
            },
            "access_steps": {
                mapping: access_steps(mapping, self.shape).to_mapping()
                for mapping in MAPPINGS
            },
            "sides": {
                timing.implementation: _rates(timing, self.shape.flops)
                for timing in self.timings
            },
            "naive_to_torch_gflops_ratio": {
                name: torch_median / self.median_ns(name) for name in (columns, rows)
            },
            "x_rows_to_x_columns_gflops_ratio": self.median_ns(columns)
            / self.median_ns(rows),
            "torch_disagreement": self.torch_disagreement,
            "agreement_bound": agreement_bound(self.shape.k),
        }


def _check(result: object, point: str, inner: int) -> float:
    if not isinstance(result, WorkloadCheck):
        raise TypeError("a workload check must return a WorkloadCheck")
    if result.naive_same_bits is not True:
        raise MatmulMetalError(f"{point}: the timed naive outputs differ in bits")
    disagreement = result.torch_disagreement
    if (
        type(disagreement) is not float
        or not math.isfinite(disagreement)
        or not 0.0 <= disagreement <= agreement_bound(inner)
    ):
        raise MatmulMetalError(
            f"{point}: torch.matmul sits {disagreement!r} from the naive output; "
            f"the agreement bound is {agreement_bound(inner)!r}"
        )
    return disagreement


def benchmark_sweep(
    backend: object,
    *,
    seed: int,
    shapes: tuple[MatmulShape, ...],
    group_size: tuple[int, int],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[MatmulPoint, ...]:
    """Time both naive kernels and ``torch.matmul`` at every sweep shape."""

    if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_POINTS:
        raise MatmulMetalError(f"a sweep holds 1 to {MAX_POINTS} shapes")
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
                KERNEL_FOR_MAPPING[mapping]: launch
                for mapping, launch in zip(MAPPINGS, launches)
            },
        )
        timings = time_sides(
            workload,
            calls_per_sample=calls_per_sample(shape.flops, target_flops_per_sample),
            warmup=warmup,
            repeats=repeats,
            clock=clock,
            budget=budget,
        )
        label = f"point {index} ({shape.m} x {shape.n} x {shape.k})"
        disagreement = _check(workload.check(), label, shape.k)
        points.append(MatmulPoint(index, shape, launches, timings, disagreement))
        del workload
    return tuple(points)


def ceiling_summary(points: tuple[MatmulPoint, ...]) -> dict[str, object]:
    """Compare each naive kernel's best median rate with the measured ceiling.

    The ceiling point is the first point with the highest median rate of
    ``torch.matmul``; each kernel's best point is found the same way.
    """

    if type(points) is not tuple or not points:
        raise MatmulMetalError("a ceiling needs at least one sweep point")

    def best(side: str) -> MatmulPoint:
        return max(points, key=lambda point: point.median_gflops(side))

    top = best(TORCH_MATMUL)
    ceiling = top.median_gflops(TORCH_MATMUL)
    naive = {}
    for name in KERNELS:
        point = best(name)
        rate = point.median_gflops(name)
        naive[name] = {
            "best_median_gflops_per_second": rate,
            "best_index": point.index,
            "fraction_of_measured_ceiling": rate / ceiling,
        }
    return {
        "definition": CEILING,
        "measured_ceiling_gflops_per_second": ceiling,
        "ceiling_index": top.index,
        "naive": naive,
    }
