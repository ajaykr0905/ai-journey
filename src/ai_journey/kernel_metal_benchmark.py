"""Synchronized wall-clock timing of each Metal kernel against PyTorch on MPS.

A GPU launch returns once its command is queued, so a host clock read right
after it would time almost nothing. Each timed sample therefore calls
``synchronize`` before its first clock reading, so no earlier work is still
running, launches ``calls_per_sample`` back-to-back calls, and calls
``synchronize`` again before its second reading, so the sample covers their
completion. Each implementation first runs ``warmup`` untimed samples, which
absorb one-time costs such as PyTorch building its MPS graphs. Timed samples
then alternate between the Metal kernel and PyTorch's equivalent, and the
order flips every repeat, so slow drift in clocks or load favors neither
side. After the last sample, the latest outputs of both sides must agree
within the kernel's bound.

A sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``, at most ``MAX_CALLS_PER_SAMPLE``, so a small
point is not lost in the cost of synchronizing. Effective bandwidth divides
those bytes by the median sample: 12 bytes per vector element and 4 per
pixel, what each kernel must read and write. PyTorch's grayscale runs as ten
eager operations with temporaries and moves more bytes than it is credited
with; the shared numerator makes the two rates a direct comparison of time.

Only the host timer ``time.perf_counter_ns`` makes a measurement; any other
clock is named ``injected`` and its durations are synthetic. These are
host-specific Apple GPU (Metal) timings; nothing here measures CUDA or
NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat

from .kernel_metal_correctness import METAL_KERNEL, TORCH
from .kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    image_launch,
    require_integer,
    vector_launch,
)
from .kernel_metal_reference import uniform_rgb_image, vector_add_inputs
from .kernel_metal_source import KERNELS, RGB_TO_GRAY, VECTOR_ADD
from .memory_access_benchmark import (
    MAX_REPEATS,
    MAX_WARMUP,
    TimingSummary,
    summarize_timings,
    timer_name,
)

MIN_REPEATS = 3
MAX_CALLS_PER_SAMPLE = 4096
MIN_TARGET_BYTES = 1 << 10
MAX_TARGET_BYTES = 1 << 30
MAX_RUNTIME_SECONDS = 600
MAX_POINTS = 16
# Benchmark inputs use case indices from here up, apart from correctness.
BENCHMARK_INDEX_OFFSET = 32
NS_PER_SECOND = 1_000_000_000
SIDES = (METAL_KERNEL, TORCH)


def calls_per_sample(bytes_per_call: int, target_bytes: int) -> int:
    """Return the fewest whole calls whose compulsory bytes reach the target.

    The batch size follows from the accounting alone, so a verifier can
    recompute it from the report.
    """

    per_call = require_integer(bytes_per_call, "bytes_per_call", 1, 1 << 62)
    target = require_integer(
        target_bytes, "target_bytes_per_sample", MIN_TARGET_BYTES, MAX_TARGET_BYTES
    )
    calls = max(1, -(-target // per_call))
    if calls > MAX_CALLS_PER_SAMPLE:
        raise KernelMetalError(
            f"a sample would need {calls} calls; at most {MAX_CALLS_PER_SAMPLE} "
            "are allowed"
        )
    return calls


class _Reader:
    """Read a clock and require nonnegative, nondecreasing integer nanoseconds."""

    def __init__(self, clock: Callable[[], int], label: str) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._label = label
        self._last: int | None = None

    def __call__(self) -> int:
        value = self._clock()
        if type(value) is not int or value < 0:
            raise KernelMetalError(
                f"{self._label} clock must return nonnegative integer nanoseconds"
            )
        if self._last is not None and value < self._last:
            raise KernelMetalError(f"{self._label} clock must be monotonic")
        self._last = value
        return value


class RunBudget:
    """A wall-clock limit for one run, read from the clock that times the run.

    It is checked only between samples, so it never splits a timed region,
    and a run stops at the first check after its budget runs out.
    """

    def __init__(self, max_runtime_seconds: int, clock: Callable[[], int]) -> None:
        self.max_runtime_seconds = require_integer(
            max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )
        self.limit_ns = self.max_runtime_seconds * NS_PER_SECOND
        self._read = _Reader(clock, "run")
        self.started_ns = self._read()
        self.elapsed_ns = 0

    def check(self) -> int:
        """Record the elapsed run time and stop the run once it is over budget."""

        self.elapsed_ns = self._read() - self.started_ns
        if self.elapsed_ns > self.limit_ns:
            raise KernelMetalError(
                f"run exceeded max_runtime_seconds={self.max_runtime_seconds} "
                f"after {self.elapsed_ns} ns"
            )
        return self.elapsed_ns


@dataclass(frozen=True)
class DeviceTiming:
    """Raw sample durations of one implementation, with the batch behind them."""

    implementation: str
    calls_per_sample: int
    warmup: int
    repeats: int
    samples_ns: tuple[int, ...]
    timer: str

    @property
    def summary(self) -> TimingSummary:
        """Nearest-rank min, quartiles, median, max and IQR of the samples."""

        return summarize_timings(self.samples_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples beside their statistics, so both can be checked."""

        return {
            "implementation": self.implementation,
            "calls_per_sample": self.calls_per_sample,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "samples_ns": list(self.samples_ns),
            "timing": self.summary.to_mapping(),
        }


def time_pair(
    workload: object,
    *,
    calls_per_sample: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[DeviceTiming, DeviceTiming]:
    """Time the Metal kernel and PyTorch's equivalent in alternating samples.

    Returns the Metal timing first. Every sample, untimed or timed, starts and
    ends with ``workload.synchronize()``.
    """

    calls = require_integer(
        calls_per_sample, "calls_per_sample", 1, MAX_CALLS_PER_SAMPLE
    )
    warm = require_integer(warmup, "warmup", 0, MAX_WARMUP)
    count = require_integer(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    sides = {METAL_KERNEL: workload.metal, TORCH: workload.torch}
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
        for name in SIDES if index % 2 == 0 else SIDES[::-1]:
            synchronize()
            started = read()
            batch(sides[name])
            synchronize()
            elapsed = read() - started
            if elapsed <= 0:
                raise KernelMetalError(
                    "benchmark clock did not advance across a sample"
                )
            samples[name].append(elapsed)
            checkpoint()
    timer = timer_name(clock)
    metal, torch = (
        DeviceTiming(name, calls, warm, count, tuple(samples[name]), timer)
        for name in SIDES
    )
    return metal, torch


def _rates(timing: DeviceTiming, bytes_per_call: int) -> dict[str, object]:
    summary = timing.summary
    calls = timing.calls_per_sample
    moved = bytes_per_call * calls
    return {
        **timing.to_mapping(),
        "bytes_per_sample": moved,
        "median_ns_per_call": summary.median_ns / calls,
        "iqr_ns_per_call": summary.iqr_ns / calls,
        "median_gb_per_second": moved / summary.median_ns,
        "best_gb_per_second": moved / summary.min_ns,
    }


@dataclass(frozen=True)
class BenchmarkPoint:
    """One sweep point: both implementations timed on the same device buffers."""

    kernel: str
    launch: CheckedLaunch
    metal: DeviceTiming
    torch: DeviceTiming

    @property
    def bytes_per_call(self) -> int:
        """Compulsory bytes one call reads and writes, for both sides."""

        return KERNELS[self.kernel].bytes_per_item * self.launch.geometry.active_threads

    @property
    def torch_to_metal_time_ratio(self) -> float:
        """PyTorch's median sample over the Metal kernel's; above 1, Metal won."""

        return self.torch.summary.median_ns / self.metal.summary.median_ns

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics and derived rates."""

        extent = self.launch.geometry.extent
        shape = (
            {"length": extent[0]}
            if self.kernel == VECTOR_ADD
            else {"height": extent[1], "width": extent[0]}
        )
        return {
            "kernel": self.kernel,
            **shape,
            "launch": self.launch.to_mapping(),
            "bytes_per_call": self.bytes_per_call,
            METAL_KERNEL: _rates(self.metal, self.bytes_per_call),
            TORCH: _rates(self.torch, self.bytes_per_call),
            "torch_to_metal_time_ratio": self.torch_to_metal_time_ratio,
        }


def _measure(
    kernel: str,
    workload: object,
    launch: CheckedLaunch,
    *,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int],
    budget: RunBudget | None,
) -> BenchmarkPoint:
    bytes_per_call = KERNELS[kernel].bytes_per_item * launch.geometry.active_threads
    metal, torch = time_pair(
        workload,
        calls_per_sample=calls_per_sample(bytes_per_call, target_bytes_per_sample),
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    if not workload.check():
        raise KernelMetalError(
            f"{kernel} at {list(launch.geometry.extent)}: the timed Metal and "
            "PyTorch outputs disagree"
        )
    return BenchmarkPoint(kernel=kernel, launch=launch, metal=metal, torch=torch)


def _points(values: object, label: str) -> tuple[object, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= MAX_POINTS:
        raise KernelMetalError(f"{label} must be a tuple of 1 to {MAX_POINTS} points")
    return values


def benchmark_vector_add(
    backend: object,
    *,
    seed: int,
    lengths: tuple[int, ...],
    group_size: int,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[BenchmarkPoint, ...]:
    """Time the Metal vector add against ``torch.add`` at each length."""

    points = []
    for index, length in enumerate(_points(lengths, "vector_add lengths")):
        first, second = vector_add_inputs(seed, BENCHMARK_INDEX_OFFSET + index, length)
        launch = CheckedLaunch(
            vector_launch(length, group_size), backend.limits[VECTOR_ADD]
        )
        workload = backend.vector_add_workload(first, second, launch)
        points.append(
            _measure(
                VECTOR_ADD,
                workload,
                launch,
                warmup=warmup,
                repeats=repeats,
                target_bytes_per_sample=target_bytes_per_sample,
                clock=clock,
                budget=budget,
            )
        )
        del workload
    return tuple(points)


def benchmark_rgb_to_gray(
    backend: object,
    *,
    seed: int,
    shapes: tuple[tuple[int, int], ...],
    group_size: tuple[int, int],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[BenchmarkPoint, ...]:
    """Time the Metal grayscale kernel against PyTorch's eager equivalent."""

    points = []
    for index, shape in enumerate(_points(shapes, "rgb_to_gray shapes")):
        if type(shape) is not tuple or len(shape) != 2:
            raise KernelMetalError("an image shape must be a (height, width) tuple")
        height, width = shape
        image = uniform_rgb_image(seed, BENCHMARK_INDEX_OFFSET + index, height, width)
        launch = CheckedLaunch(
            image_launch(height, width, group_size), backend.limits[RGB_TO_GRAY]
        )
        workload = backend.rgb_to_gray_workload(image, launch)
        points.append(
            _measure(
                RGB_TO_GRAY,
                workload,
                launch,
                warmup=warmup,
                repeats=repeats,
                target_bytes_per_sample=target_bytes_per_sample,
                clock=clock,
                budget=budget,
            )
        )
        del workload
    return tuple(points)
