"""Time every admitted shape of one sweep point in interleaved, synchronized samples.

A sample calls ``synchronize`` before its first clock reading, launches
``calls_per_sample`` calls with one shape, and calls ``synchronize`` again
before its second reading, as the Day 73 benchmark does. Each shape first
runs ``warmup`` untimed samples. The timed samples then take turns: repeat
``r`` starts with shape ``r mod S`` and runs each of the ``S`` shapes once in
order. Each shape thus takes each position equally often when the repeat
count is a multiple of ``S``, so slow drift in clocks, heat or load spreads
over every shape instead of favoring the shapes that run first. One batch
size and one set of input buffers per point keep the shapes comparable.

A batch holds the fewest whole calls that reach the sample target: ``2 m n
k`` FLOPs per matmul call, and 12 compulsory bytes per vector element, with
the Day 73 ``calls_per_sample``. A rate divides the work of a sample by its
median sample: GFLOP/s for the matmul and decimal GB/s for the vector add.
The ranking orders shapes by median sample time, and equal medians keep the
config order. ``separated`` is true only when the interquartile ranges of the
fastest and the slowest shape do not overlap. Otherwise the spread of the
point is inside the run-to-run noise of its samples.

Only ``time.perf_counter_ns`` makes a measurement; any other clock is named
``injected``. These are host-specific Apple GPU (Metal) timings. Nothing here
measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat

import numpy

from .kernel_metal_benchmark import (
    BENCHMARK_INDEX_OFFSET,
    MAX_CALLS_PER_SAMPLE,
    MIN_REPEATS,
    RunBudget,
    _Reader,
    calls_per_sample,
)
from .kernel_metal_geometry import require_integer, vector_launch
from .kernel_metal_reference import (
    array_sha256,
    vector_add_inputs,
    vector_add_reference,
)
from .kernel_metal_source import KERNELS, VECTOR_ADD
from .memory_access_benchmark import (
    MAX_REPEATS,
    MAX_WARMUP,
    TimingSummary,
    gflops_per_second,
    summarize_timings,
    timer_name,
)
from .metal_launch_sweep_correctness import (
    MatmulAccuracy,
    _admitted,
    matmul_accuracy,
    matmul_inputs,
)
from .metal_launch_sweep_shapes import (
    MetalLaunchSweepError,
    ShapeAdmission,
    checked_launch,
    matmul_launch,
)
from .metal_launch_sweep_source import MATMUL

MIN_TARGET_FLOPS = 1 << 20
MAX_TARGET_FLOPS = 1 << 40
VECTOR_BYTES_PER_ELEMENT = KERNELS[VECTOR_ADD].bytes_per_item


def matmul_flops(m: int, n: int, k: int) -> int:
    """Count one multiply and one add per inner step of every output element."""

    return 2 * m * n * k


def matmul_calls_per_sample(flops_per_call: int, target_flops: int) -> int:
    """Return the fewest whole matmul calls whose FLOPs reach the target."""

    per_call = require_integer(flops_per_call, "flops_per_call", 1, 1 << 62)
    target = require_integer(
        target_flops, "target_flops_per_sample", MIN_TARGET_FLOPS, MAX_TARGET_FLOPS
    )
    calls = -(-target // per_call)
    if calls > MAX_CALLS_PER_SAMPLE:
        raise MetalLaunchSweepError(
            f"a sample would need {calls} calls; at most {MAX_CALLS_PER_SAMPLE} "
            "are allowed"
        )
    return calls


def time_shapes(
    workload: object,
    *,
    calls_per_sample: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Time each call of ``workload`` in rotating turns; return its samples in order.

    Every sample, untimed or timed, starts and ends with
    ``workload.synchronize()``. The budget is checked only between samples.
    """

    calls = require_integer(
        calls_per_sample, "calls_per_sample", 1, MAX_CALLS_PER_SAMPLE
    )
    warm = require_integer(warmup, "warmup", 0, MAX_WARMUP)
    count = require_integer(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    shapes = workload.calls
    if type(shapes) is not tuple or not shapes:
        raise MetalLaunchSweepError("a workload needs one call per shape")
    synchronize = workload.synchronize
    read = _Reader(clock, "benchmark")

    def batch(call: Callable[[], object]) -> None:
        for _ in repeat(None, calls):
            call()

    def checkpoint() -> None:
        if budget is not None:
            budget.check()

    for call in shapes:
        for _ in range(warm):
            synchronize()
            batch(call)
            synchronize()
            checkpoint()
    samples: list[list[int]] = [[] for _ in shapes]
    for index in range(count):
        for offset in range(len(shapes)):
            position = (index + offset) % len(shapes)
            synchronize()
            started = read()
            batch(shapes[position])
            synchronize()
            elapsed = read() - started
            if elapsed <= 0:
                raise MetalLaunchSweepError(
                    "benchmark clock did not advance across a sample"
                )
            samples[position].append(elapsed)
            checkpoint()
    return tuple(tuple(series) for series in samples)


@dataclass(frozen=True)
class ShapeTiming:
    """The raw samples of one admitted shape at one sweep point."""

    admission: ShapeAdmission
    samples_ns: tuple[int, ...]

    @property
    def summary(self) -> TimingSummary:
        """Nearest-rank min, quartiles, median, max and IQR of the samples."""

        return summarize_timings(self.samples_ns)


@dataclass(frozen=True)
class SweepPoint:
    """Every admitted shape timed on the same buffers at one extent."""

    kernel: str
    extent: tuple[tuple[str, int], ...]
    work_per_call: int
    calls_per_sample: int
    warmup: int
    repeats: int
    timer: str
    timings: tuple[ShapeTiming, ...]
    output_sha256: str
    accuracy: MatmulAccuracy | None

    def _order(self) -> list[int]:
        return sorted(
            range(len(self.timings)),
            key=lambda position: (self.timings[position].summary.median_ns, position),
        )

    @property
    def ranking(self) -> tuple[ShapeTiming, ...]:
        """Shapes from the fastest median sample to the slowest; ties keep order."""

        return tuple(self.timings[position] for position in self._order())

    def _rate(self, samples_ns: int) -> float:
        work = self.work_per_call * self.calls_per_sample
        if self.kernel == MATMUL:
            return gflops_per_second(work, samples_ns)
        return work / samples_ns

    def to_mapping(self) -> dict[str, object]:
        """Return raw samples beside the statistics, rates and ranking they give."""

        rate = "gflops_per_second" if self.kernel == MATMUL else "gb_per_second"
        order = self._order()
        rank = {position: place for place, position in enumerate(order, 1)}
        fastest = self.timings[order[0]].summary
        slowest = self.timings[order[-1]].summary
        shapes = []
        for position, timing in enumerate(self.timings):
            summary = timing.summary
            admission = timing.admission
            shapes.append(
                {
                    "group_size": list(admission.group_size),
                    "threads_per_group": admission.threads_per_group,
                    **admission.footprint.to_mapping(),
                    "samples_ns": list(timing.samples_ns),
                    "timing": summary.to_mapping(),
                    "median_ns_per_call": summary.median_ns / self.calls_per_sample,
                    "iqr_ns_per_call": summary.iqr_ns / self.calls_per_sample,
                    f"median_{rate}": self._rate(summary.median_ns),
                    f"best_{rate}": self._rate(summary.min_ns),
                    "rank": rank[position],
                    "relative_to_fastest": summary.median_ns / fastest.median_ns,
                }
            )
        mapping = {
            "kernel": self.kernel,
            **dict(self.extent),
            "work_per_call": self.work_per_call,
            "calls_per_sample": self.calls_per_sample,
            "work_per_sample": self.work_per_call * self.calls_per_sample,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "shapes": shapes,
            "ranking": [list(timing.admission.group_size) for timing in self.ranking],
            "spread": slowest.median_ns / fastest.median_ns,
            "separated": fastest.p75_ns < slowest.p25_ns,
            "output_sha256": self.output_sha256,
        }
        if self.accuracy is not None:
            mapping["accuracy"] = self.accuracy.to_mapping()
        return mapping


def measure_matmul_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: tuple[int, int, int],
    admissions: tuple[ShapeAdmission, ...],
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> SweepPoint:
    """Time every admitted shape of the matmul at one size, then check C.

    After the last sample, every shape's output must be bit-identical, and
    the first must meet the float32 bound of the float64 product.
    """

    m, n, k = shape
    a, b = matmul_inputs(seed, BENCHMARK_INDEX_OFFSET + index, m, n, k)
    chosen = _admitted(admissions)
    launches = tuple(
        checked_launch(item, matmul_launch(m, n, item.group_size)) for item in chosen
    )
    flops = matmul_flops(m, n, k)
    calls = matmul_calls_per_sample(flops, target_flops_per_sample)
    workload = backend.matmul_workload(a, b, launches)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"matmul point {m}x{n}x{k}"
    if not workload.outputs_agree():
        raise MetalLaunchSweepError(f"{label}: the timed outputs differ between shapes")
    accuracy = matmul_accuracy(a, b, workload.output(), label)
    return SweepPoint(
        kernel=MATMUL,
        extent=(("m", m), ("n", n), ("k", k)),
        work_per_call=flops,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        timer=timer_name(clock),
        timings=tuple(ShapeTiming(*pair) for pair in zip(chosen, samples)),
        output_sha256=accuracy.output_sha256,
        accuracy=accuracy,
    )


def measure_vector_point(
    backend: object,
    *,
    seed: int,
    index: int,
    length: int,
    admissions: tuple[ShapeAdmission, ...],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> SweepPoint:
    """Time every admitted group size of the Day 73 vector add at one length.

    After the last sample, every size's output must equal the float32 sum
    bit for bit.
    """

    first, second = vector_add_inputs(seed, BENCHMARK_INDEX_OFFSET + index, length)
    chosen = _admitted(admissions)
    launches = tuple(
        checked_launch(item, vector_launch(length, item.group_size[0]))
        for item in chosen
    )
    moved = VECTOR_BYTES_PER_ELEMENT * length
    calls = calls_per_sample(moved, target_bytes_per_sample)
    workload = backend.vector_workload(first, second, launches)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"vector point {length}"
    if not workload.outputs_agree():
        raise MetalLaunchSweepError(f"{label}: the timed outputs differ between sizes")
    reference = vector_add_reference(first, second)
    output = workload.output()
    if not (
        isinstance(output, numpy.ndarray)
        and output.dtype == reference.dtype
        and output.shape == reference.shape
        and numpy.array_equal(output.view("u4"), reference.view("u4"))
    ):
        raise MetalLaunchSweepError(f"{label}: the output differs from the float32 sum")
    return SweepPoint(
        kernel=VECTOR_ADD,
        extent=(("length", length),),
        work_per_call=moved,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        timer=timer_name(clock),
        timings=tuple(ShapeTiming(*pair) for pair in zip(chosen, samples)),
        output_sha256=array_sha256(reference),
        accuracy=None,
    )
