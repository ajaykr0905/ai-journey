"""Bounded wall-clock timing core for CPU measurements on this host.

Timings come from an injectable integer nanosecond clock that defaults to
``time.perf_counter_ns``. Warmup calls run but are never timed. Each measured
repeat times exactly one operation call between two clock readings, and
result inspection runs after the clock stops, for warmup and measured calls
alike. Percentiles use the nearest-rank method without interpolation
(``serving_analysis.nearest_rank_percentile``).

These are measurements of whatever host runs them; nothing here measures or
predicts GPU behavior.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .memory_access_pattern import MemoryAccessError
from .serving_analysis import nearest_rank_percentile

TIMER = "time.perf_counter_ns"
MAX_REPEATS = 50
MAX_WARMUP = 10
PERCENTILE_METHOD = (
    "nearest rank: sorted samples[ceil(p * n / 100) - 1] for p in 25, 50 and 75; "
    "median is p50 and IQR is p75 - p25; no interpolation"
)

Result = TypeVar("Result")


@dataclass(frozen=True)
class TimingSummary:
    """Nearest-rank statistics of positive per-repeat durations in nanoseconds."""

    count: int
    min_ns: int
    p25_ns: int
    median_ns: int
    p75_ns: int
    max_ns: int

    @property
    def iqr_ns(self) -> int:
        return self.p75_ns - self.p25_ns

    def to_mapping(self) -> dict[str, int]:
        return {
            "count": self.count,
            "min_ns": self.min_ns,
            "p25_ns": self.p25_ns,
            "median_ns": self.median_ns,
            "p75_ns": self.p75_ns,
            "max_ns": self.max_ns,
            "iqr_ns": self.iqr_ns,
        }


def summarize_timings(samples_ns: tuple[int, ...]) -> TimingSummary:
    """Summarize measured durations without interpolating between samples."""

    if type(samples_ns) is not tuple or not samples_ns:
        raise MemoryAccessError("samples_ns must be a nonempty tuple")
    if any(type(sample) is not int or sample <= 0 for sample in samples_ns):
        raise MemoryAccessError("samples_ns must hold positive integer nanoseconds")
    return TimingSummary(
        count=len(samples_ns),
        min_ns=min(samples_ns),
        p25_ns=nearest_rank_percentile(samples_ns, 25),
        median_ns=nearest_rank_percentile(samples_ns, 50),
        p75_ns=nearest_rank_percentile(samples_ns, 75),
        max_ns=max(samples_ns),
    )


def gflops_per_second(flops: int, elapsed_ns: int) -> float:
    """Convert FLOPs over a duration to GFLOP/s (one FLOP per ns is 1 GFLOP/s)."""

    for name, value in (("flops", flops), ("elapsed_ns", elapsed_ns)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise MemoryAccessError(f"{name} must be positive")
    return flops / elapsed_ns


def _count(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise MemoryAccessError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def time_repeats(
    operation: Callable[[], Result],
    *,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    inspect: Callable[[Result], None] | None = None,
) -> tuple[int, ...]:
    """Time ``repeats`` calls after ``warmup`` untimed calls.

    The clock must return nondecreasing nonnegative integers and must advance
    across every timed call; a clock too coarse for the operation is an error
    rather than a zero-duration sample.
    """

    warm = _count(warmup, "warmup", 0, MAX_WARMUP)
    count = _count(repeats, "repeats", 1, MAX_REPEATS)
    if not callable(operation) or not callable(clock):
        raise TypeError("operation and clock must be callable")
    if inspect is not None and not callable(inspect):
        raise TypeError("inspect must be callable")
    previous: int | None = None

    def read() -> int:
        nonlocal previous
        value = clock()
        if type(value) is not int or value < 0:
            raise MemoryAccessError(
                "benchmark clock must return nonnegative integer nanoseconds"
            )
        if previous is not None and value < previous:
            raise MemoryAccessError("benchmark clock must be monotonic")
        previous = value
        return value

    for _ in range(warm):
        result = operation()
        if inspect is not None:
            inspect(result)
    samples = []
    for _ in range(count):
        started = read()
        result = operation()
        elapsed = read() - started
        if elapsed <= 0:
            raise MemoryAccessError("benchmark clock did not advance across a call")
        samples.append(elapsed)
        if inspect is not None:
            inspect(result)
    return tuple(samples)
