"""Batched CPU wall-clock samples under a run budget, on the shared timing core.

One sample times ``calls_per_sample`` back-to-back calls of a kernel between
two clock readings. A cache-sized kernel call lasts around a microsecond, too
short to time alone against the clock's resolution and the cost of reading
it, so each sample is sized to move a target number of accounted bytes: the
fewest whole calls that reach it. ``memory_access_benchmark.time_repeats``
runs the untimed warmups and the timed repeats, and its nearest-rank
statistics summarize them. This module adds the batching, a run budget
checked after every warmup and timed sample (outside the timed region, so a
run overruns its budget by at most one sample), and rates divided once from
exact integers.

Durations describe whatever host runs them; nothing here measures a GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat

from .memory_access_benchmark import (
    MAX_REPEATS,
    MAX_WARMUP,
    TimingSummary,
    summarize_timings,
    time_repeats,
    timer_name,
)
from .memory_access_pattern import MemoryAccessError
from .memory_hierarchy_stream import MemoryHierarchyError, require_integer

MIN_REPEATS = 3
MAX_CALLS_PER_SAMPLE = 1 << 16
MIN_TARGET_BYTES = 1 << 10
MAX_TARGET_BYTES = 1 << 30
MAX_RUNTIME_SECONDS = 300
NS_PER_SECOND = 1_000_000_000
_MAX_COUNT = 1 << 62


def calls_per_sample(bytes_per_call: int, target_bytes: int) -> int:
    """Return the fewest whole calls whose accounted bytes reach the target.

    Sizing samples by bytes keeps every sweep point long enough to time and
    makes the batch size a function of the accounting alone, so a verifier
    can recompute it from the report.
    """

    per_call = require_integer(bytes_per_call, "bytes_per_call", 1, _MAX_COUNT)
    target = require_integer(
        target_bytes, "target_bytes_per_sample", MIN_TARGET_BYTES, MAX_TARGET_BYTES
    )
    calls = max(1, -(-target // per_call))
    if calls > MAX_CALLS_PER_SAMPLE:
        raise MemoryHierarchyError(
            f"a sample would need {calls} calls; at most {MAX_CALLS_PER_SAMPLE} "
            "are allowed"
        )
    return calls


class RunBudget:
    """A wall-clock limit for one run, read from the clock that times the run.

    Sharing the timing clock lets tests drive the budget with an injected
    clock; checking it only between samples leaves every timed region intact.
    """

    def __init__(self, max_runtime_seconds: int, clock: Callable[[], int]) -> None:
        self.max_runtime_seconds = require_integer(
            max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.limit_ns = self.max_runtime_seconds * NS_PER_SECOND
        self._clock = clock
        self._last: int | None = None
        self.started_ns = self._read()
        self.elapsed_ns = 0

    def _read(self) -> int:
        value = self._clock()
        if type(value) is not int or value < 0:
            raise MemoryHierarchyError(
                "run clock must return nonnegative integer nanoseconds"
            )
        if self._last is not None and value < self._last:
            raise MemoryHierarchyError("run clock must be monotonic")
        self._last = value
        return value

    def check(self, _result: object = None) -> int:
        """Record the elapsed run time and stop the run once it is over budget.

        Accepts and ignores a result so it can serve as the timing core's
        post-sample ``inspect`` hook.
        """

        self.elapsed_ns = self._read() - self.started_ns
        if self.elapsed_ns > self.limit_ns:
            raise MemoryHierarchyError(
                f"run exceeded max_runtime_seconds={self.max_runtime_seconds} "
                f"after {self.elapsed_ns} ns"
            )
        return self.elapsed_ns


@dataclass(frozen=True)
class BatchTiming:
    """Raw batched sample durations, the batch size and the timer behind them."""

    calls_per_sample: int
    warmup: int
    repeats: int
    samples_ns: tuple[int, ...]
    timer: str

    @property
    def summary(self) -> TimingSummary:
        """Nearest-rank min, quartiles, median and IQR of the raw samples."""

        return summarize_timings(self.samples_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return raw samples beside their statistics so both can be rechecked."""

        return {
            "calls_per_sample": self.calls_per_sample,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "samples_ns": list(self.samples_ns),
            "timing": self.summary.to_mapping(),
        }


def time_batches(
    call: Callable[[], object],
    *,
    calls_per_sample: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> BatchTiming:
    """Time ``repeats`` batches of calls after ``warmup`` untimed batches.

    Warmup batches are the same batches, so a cache-sized working set is
    resident and every page has been touched before the first timed sample.
    Clock contract violations surface as ``MemoryHierarchyError``.
    """

    calls = require_integer(
        calls_per_sample, "calls_per_sample", 1, MAX_CALLS_PER_SAMPLE
    )
    warm = require_integer(warmup, "warmup", 0, MAX_WARMUP)
    count = require_integer(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if not callable(call):
        raise TypeError("call must be callable")
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    if calls == 1:
        batch = call
    else:

        def batch() -> None:
            for _ in repeat(None, calls):
                call()

    try:
        samples = time_repeats(
            batch,
            warmup=warm,
            repeats=count,
            clock=clock,
            inspect=None if budget is None else budget.check,
        )
    except MemoryAccessError as error:
        raise MemoryHierarchyError(str(error)) from error
    return BatchTiming(
        calls_per_sample=calls,
        warmup=warm,
        repeats=count,
        samples_ns=samples,
        timer=timer_name(clock),
    )


def per_nanosecond(count: int, elapsed_ns: int) -> float:
    """Return ``count`` per nanosecond: decimal GB/s for bytes, GFLOP/s for FLOPs.

    One division of exact integers gives the builder and an independent
    verifier the identical float.
    """

    total = require_integer(count, "count", 0, _MAX_COUNT)
    elapsed = require_integer(elapsed_ns, "elapsed_ns", 1, _MAX_COUNT)
    return total / elapsed


def per_second(count: int, elapsed_ns: int) -> float:
    """Return ``count`` per second, the unit ``roofline.RooflineHardware`` takes.

    The integer product is exact, so the single division rounds once.
    """

    total = require_integer(count, "count", 0, _MAX_COUNT)
    elapsed = require_integer(elapsed_ns, "elapsed_ns", 1, _MAX_COUNT)
    return total * NS_PER_SECOND / elapsed
