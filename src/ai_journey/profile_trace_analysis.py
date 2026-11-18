"""Torch-free timeline analysis of one parsed ``torch.profiler`` trace.

All times are exact integer nanoseconds from ``profile_trace_events``. The
self time of a span is its duration minus the durations of its direct
children on the same thread, so time spent inside a nested call counts once,
in the innermost span. Per-op figures add the spans that share a name.

The profiler's own recording-window span (category ``Trace``) covers the
whole capture on a track of its own; it is not an op and is left out.

With ``ProfilerActivity.CPU`` alone, every span is host time: the Python
and C++ work of dispatch, and the time a host thread waits in a
synchronize call. No span here is a GPU kernel's own time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .profile_trace_events import (
    NestedSpan,
    ParsedTrace,
    ProfileTraceError,
    require_integer,
)

WINDOW_CATEGORY = "Trace"
MAX_TOP_N = 64


def is_op(item: NestedSpan) -> bool:
    """Return whether a span is an op and not the profiler's recording window."""

    return item.span.category != WINDOW_CATEGORY


def self_times(trace: ParsedTrace) -> tuple[int, ...]:
    """Return the self time of each span in ``trace.nested``, in that order.

    Nesting was checked at parse time, so the children of a span never add
    up to more than the span itself.
    """

    if not isinstance(trace, ParsedTrace):
        raise TypeError("trace must be a ParsedTrace")
    remaining = [item.span.duration_ns for item in trace.nested]
    for item in trace.nested:
        if item.parent is not None:
            remaining[item.parent] -= item.span.duration_ns
    if any(value < 0 for value in remaining):
        raise ProfileTraceError("children outlast their parent span")
    return tuple(remaining)


@dataclass(frozen=True)
class OpStats:
    """Every span of one name: calls, total and self time, and the longest call."""

    name: str
    calls: int
    total_ns: int
    self_ns: int
    max_ns: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ProfileTraceError("an op needs a non-empty name")
        require_integer(self.calls, "calls", 1, 1 << 62)
        for value, label in (
            (self.total_ns, "total_ns"),
            (self.self_ns, "self_ns"),
            (self.max_ns, "max_ns"),
        ):
            require_integer(value, label, 0, 1 << 62)
        if not self.self_ns <= self.total_ns or not self.max_ns <= self.total_ns:
            raise ProfileTraceError(f"op {self.name!r} has inconsistent times")

    def to_mapping(self) -> dict[str, object]:
        """Return the stats as a report records them."""

        return {
            "name": self.name,
            "calls": self.calls,
            "total_ns": self.total_ns,
            "self_ns": self.self_ns,
            "max_ns": self.max_ns,
        }


def op_stats(trace: ParsedTrace) -> tuple[OpStats, ...]:
    """Add the spans of each name into one ``OpStats``, sorted by name.

    A recursive op nested in itself adds its outer and inner totals, as the
    torch table does; its self time still counts each nanosecond once.
    """

    selves = self_times(trace)
    rows: dict[str, list[int]] = {}
    for item, own in zip(trace.nested, selves):
        if not is_op(item):
            continue
        row = rows.setdefault(item.span.name, [0, 0, 0, 0])
        row[0] += 1
        row[1] += item.span.duration_ns
        row[2] += own
        row[3] = max(row[3], item.span.duration_ns)
    return tuple(OpStats(name, *rows[name]) for name in sorted(rows))


def top_ops(stats: tuple[OpStats, ...], count: int) -> tuple[OpStats, ...]:
    """Return the ``count`` ops with the most self time, ties broken by name.

    Self time ranks the ops because total time counts a parent and its
    children both, so the outermost wrapper would always come first.
    """

    require_integer(count, "top_n", 1, MAX_TOP_N)
    if type(stats) is not tuple or not all(isinstance(row, OpStats) for row in stats):
        raise TypeError("stats must be a tuple of OpStats")
    ranked = sorted(stats, key=lambda row: (-row.self_ns, row.name))
    return tuple(ranked[:count])


def origin_ns(trace: ParsedTrace) -> int:
    """Return the start of the earliest op, the zero of every offset in a report.

    Offsets keep a report free of the host's absolute clock values.
    """

    starts = [item.span.start_ns for item in trace.nested if is_op(item)]
    if not starts:
        raise ProfileTraceError("trace holds no op spans")
    return min(starts)


@dataclass(frozen=True)
class CallGap:
    """The idle host time on one thread between two consecutive top-level calls."""

    before: str
    after: str
    offset_ns: int
    gap_ns: int

    def to_mapping(self) -> dict[str, object]:
        """Return the gap as a report records it, ``offset_ns`` at its start."""

        return {
            "before": self.before,
            "after": self.after,
            "offset_ns": self.offset_ns,
            "gap_ns": self.gap_ns,
        }


@dataclass(frozen=True)
class ThreadGaps:
    """The top-level calls of one thread and the gaps between them.

    ``busy_ns`` adds the top-level durations and ``extent_ns`` runs from the
    first start to the last end, so ``extent_ns - busy_ns`` is the total gap.
    """

    pid: int | str
    tid: int | str
    calls: int
    busy_ns: int
    extent_ns: int
    gaps: tuple[CallGap, ...]

    @property
    def total_gap_ns(self) -> int:
        """All idle time between the top-level calls of this thread."""

        return sum(gap.gap_ns for gap in self.gaps)

    def largest(self, count: int) -> tuple[CallGap, ...]:
        """Return the ``count`` longest gaps, the earliest first among equals."""

        require_integer(count, "gap count", 1, MAX_TOP_N)
        ranked = sorted(self.gaps, key=lambda gap: (-gap.gap_ns, gap.offset_ns))
        return tuple(ranked[:count])

    def to_mapping(self, count: int) -> dict[str, object]:
        """Return the thread's gap summary with its ``count`` longest gaps."""

        return {
            "pid": self.pid,
            "tid": self.tid,
            "top_level_calls": self.calls,
            "busy_ns": self.busy_ns,
            "extent_ns": self.extent_ns,
            "gaps": len(self.gaps),
            "total_gap_ns": self.total_gap_ns,
            "max_gap_ns": max((gap.gap_ns for gap in self.gaps), default=0),
            "largest_gaps": [gap.to_mapping() for gap in self.largest(count)],
        }


def top_level_gaps(trace: ParsedTrace) -> tuple[ThreadGaps, ...]:
    """Measure the gaps between consecutive top-level op calls on each thread.

    A gap is host time that no profiled call covers: Python between calls,
    or work the profiler does not record. ``nested`` already orders each
    thread by start, and nesting makes top-level calls disjoint, so every
    gap is zero or more.
    """

    origin = origin_ns(trace)
    threads: dict[tuple[int | str, int | str], list[NestedSpan]] = {}
    for item in trace.nested:
        if item.parent is None and is_op(item):
            threads.setdefault(item.span.thread, []).append(item)
    result = []
    for (pid, tid), calls in threads.items():
        gaps = tuple(
            CallGap(
                before=first.span.name,
                after=second.span.name,
                offset_ns=first.span.end_ns - origin,
                gap_ns=second.span.start_ns - first.span.end_ns,
            )
            for first, second in zip(calls, calls[1:])
        )
        if any(gap.gap_ns < 0 for gap in gaps):
            raise ProfileTraceError("top-level calls overlap")
        result.append(
            ThreadGaps(
                pid=pid,
                tid=tid,
                calls=len(calls),
                busy_ns=sum(item.span.duration_ns for item in calls),
                extent_ns=calls[-1].span.end_ns - calls[0].span.start_ns,
                gaps=gaps,
            )
        )
    return tuple(result)
