"""Draw one parsed trace as a plain-text timeline, one row per thread and depth.

A viewer such as Perfetto shows the same trace with zoom and color; a text
timeline fits in a terminal, a test and a report, and it is deterministic.
The columns split the op extent (the earliest op start to the latest op
end) into equal integer-nanosecond slices. Each cell shows the letter of
the span at that depth that overlaps the slice for the longest time, or
``.`` when no span at that depth touches it. Ties go to the earlier span. A
legend maps each letter to a span name, in the order of ``nested``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from .profile_trace_analysis import is_op, origin_ns
from .profile_trace_events import ParsedTrace, require_integer

MIN_WIDTH = 16
MAX_WIDTH = 240
MAX_ROWS = 8
EMPTY = "."
OVERFLOW = "#"
SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _symbols(trace: ParsedTrace, depth_limit: int) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in trace.nested:
        if is_op(item) and item.depth < depth_limit and item.span.name not in names:
            position = len(names)
            names[item.span.name] = (
                SYMBOLS[position] if position < len(SYMBOLS) else OVERFLOW
            )
    return names


def render_timeline(trace: ParsedTrace, *, width: int, depth_limit: int) -> str:
    """Return the timeline text with ``width`` columns and ``depth_limit`` rows.

    Spans deeper than ``depth_limit`` are left out, so wrapper annotations
    stay readable when an aten op nests many levels down.
    """

    if not isinstance(trace, ParsedTrace):
        raise TypeError("trace must be a ParsedTrace")
    require_integer(width, "timeline width", MIN_WIDTH, MAX_WIDTH)
    require_integer(depth_limit, "timeline depth", 1, MAX_ROWS)
    origin = origin_ns(trace)
    end = max(item.span.end_ns for item in trace.nested if is_op(item))
    extent = max(end - origin, 1)
    bounds = [origin + extent * column // width for column in range(width + 1)]
    symbols = _symbols(trace, depth_limit)
    rows: dict[tuple[object, object], list[list[tuple[int, int, str]]]] = {}
    for order, item in enumerate(trace.nested):
        if not is_op(item) or item.depth >= depth_limit:
            continue
        thread = rows.setdefault(
            item.span.thread,
            [[(0, 0, EMPTY)] * width for _ in range(depth_limit)],
        )
        cells = thread[item.depth]
        span = item.span
        first = max(bisect_right(bounds, span.start_ns) - 1, 0)
        last = min(max(bisect_left(bounds, span.end_ns) - 1, first), width - 1)
        for column in range(first, last + 1):
            low, high = bounds[column], bounds[column + 1]
            overlap = min(high, span.end_ns) - max(low, span.start_ns)
            # A zero-length span still marks the slice it falls in.
            weight = max(overlap, 1)
            if (weight, -order) > (cells[column][0], -cells[column][1]):
                cells[column] = (weight, order, symbols[span.name])
    lines = [
        f"timeline extent_ns={extent} columns={width} "
        f"ns_per_column={extent / width:.1f}"
    ]
    for (pid, tid), depths in rows.items():
        lines.append(f"thread pid={pid} tid={tid}")
        for depth, cells in enumerate(depths):
            if all(cell[2] == EMPTY for cell in cells):
                break
            lines.append(f"  d{depth} |{''.join(cell[2] for cell in cells)}|")
    lines.append("legend")
    lines.extend(f"  {symbol} {name}" for name, symbol in symbols.items())
    return "\n".join(lines) + "\n"
