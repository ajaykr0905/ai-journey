"""Time each launch on three clocks, one clock per sample, in rotating order.

Every sample holds exactly one launch and reads exactly one clock, so no
clock's instrument sits inside another clock's interval:

- event sample: drain the queue, record the start event, launch, record the
  end event, drain the queue, then read ``elapsed_time``;
- synchronized sample: drain the queue, read the host clock, launch,
  synchronize, read the host clock;
- dispatch sample: drain the queue, read the host clock, launch, read the
  host clock, then drain the queue outside the interval.

Each side first runs ``warmup`` untimed samples on each clock, which absorb
one-time costs such as building pipelines and MPS graphs. Timed samples then
rotate: repeat ``r`` visits the sides in an order shifted by ``r`` and, for
each side, the clocks in an order shifted by ``r``. Slow drift in clocks or
load then favors no side and no clock. This is the Day 73 and 74 rotation
with one more axis.

A dispatch sample covers a strict part of what a synchronized sample covers,
so a side whose dispatch median exceeds its synchronized median is a broken
run, and it is refused. Only ``time.perf_counter_ns`` makes a host
measurement; any other clock is named ``injected``. These are host-specific
Apple GPU (Metal) timings; nothing here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .gpu_timing_clocks import (
    CLOCKS,
    DISPATCH,
    EVENT,
    SYNCHRONIZED,
    ClockSamples,
    GpuTimingError,
    require_clock,
)
from .gpu_timing_config import require_count
from .kernel_metal_benchmark import MIN_REPEATS, RunBudget, _Reader
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP, timer_name

SCHEDULE = (
    "one launch and one clock per sample; per side, warmup untimed samples on "
    "each clock first; then repeat r visits the sides shifted by r and, for "
    "each side, the clocks shifted by r"
)
MAX_SIDES = 64


@dataclass(frozen=True)
class SideTiming:
    """One side's timed samples on all three clocks, with the controls behind them."""

    implementation: str
    warmup: int
    repeats: int
    clocks: tuple[ClockSamples, ...]
    timer: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str) or not self.implementation:
            raise GpuTimingError("a side needs a name")
        require_count(self.warmup, "warmup", 0, MAX_WARMUP)
        require_count(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        if (
            type(self.clocks) is not tuple
            or not all(isinstance(item, ClockSamples) for item in self.clocks)
            or tuple(item.clock for item in self.clocks) != CLOCKS
        ):
            raise GpuTimingError(f"a side needs samples of {', '.join(CLOCKS)}")
        if any(len(item.samples_ns) != self.repeats for item in self.clocks):
            raise GpuTimingError("every clock needs one sample per repeat")
        if self.median_ns(DISPATCH) > self.median_ns(SYNCHRONIZED):
            raise GpuTimingError(
                f"{self.implementation}: the dispatch median exceeds the "
                "synchronized median, which covers the dispatch"
            )

    def samples(self, clock: str) -> ClockSamples:
        """Return one clock's samples."""

        return self.clocks[CLOCKS.index(require_clock(clock))]

    def median_ns(self, clock: str) -> int:
        """Return one clock's median sample."""

        return self.samples(clock).summary.median_ns

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples of every clock beside their statistics."""

        return {
            "implementation": self.implementation,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "clocks": {item.clock: item.to_mapping() for item in self.clocks},
        }


def time_sides(
    sides: Mapping[str, Callable[[], object]],
    *,
    synchronize: Callable[[], None],
    event_pair: Callable[[], object],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[SideTiming, ...]:
    """Time every side on every clock; return one timing per side in order.

    ``event_pair`` returns an object with ``start``, ``stop`` and
    ``elapsed_ms``; each side gets its own pair.
    """

    warm = require_count(warmup, "warmup", 0, MAX_WARMUP)
    count = require_count(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    names = tuple(sides)
    if not 1 <= len(names) <= MAX_SIDES:
        raise GpuTimingError(f"a timing needs 1 to {MAX_SIDES} sides")
    pairs = {name: event_pair() for name in names}
    read = _Reader(clock, "benchmark")

    def sample(name: str, which: str) -> int | float:
        call = sides[name]
        synchronize()
        if which == EVENT:
            pair = pairs[name]
            pair.start()
            call()
            pair.stop()
            synchronize()
            return pair.elapsed_ms()
        started = read()
        call()
        if which == SYNCHRONIZED:
            synchronize()
            return read() - started
        ended = read()
        synchronize()
        return ended - started

    def checkpoint() -> None:
        if budget is not None:
            budget.check()

    for name in names:
        for which in CLOCKS:
            for _ in range(warm):
                sample(name, which)
                checkpoint()
    raw: dict[tuple[str, str], list[int | float]] = {
        (name, which): [] for name in names for which in CLOCKS
    }
    for index in range(count):
        side_shift = index % len(names)
        clock_shift = index % len(CLOCKS)
        for name in names[side_shift:] + names[:side_shift]:
            for which in CLOCKS[clock_shift:] + CLOCKS[:clock_shift]:
                value = sample(name, which)
                if which != EVENT and value <= 0:
                    raise GpuTimingError(
                        "benchmark clock did not advance across a sample"
                    )
                raw[name, which].append(value)
                checkpoint()
    timer = timer_name(clock)
    timings = []
    for name in names:
        event = ClockSamples.from_event_ms(tuple(raw[name, EVENT]))
        host = (
            ClockSamples(which, tuple(raw[name, which]))
            for which in (SYNCHRONIZED, DISPATCH)
        )
        timings.append(SideTiming(name, warm, count, (event, *host), timer))
    return tuple(timings)
