"""Strict, torch-free parser for the Chrome trace-event JSON of ``torch.profiler``.

``torch.profiler.profile.export_chrome_trace`` writes one JSON object whose
``traceEvents`` array holds complete events (``X``), instant events (``i``)
and metadata events (``M``); other writers of the same format also use
begin and end pairs (``B`` and ``E``). This module accepts exactly those
phases and refuses everything else by name, so a trace that would be read
in a different way by a viewer is never analyzed in silence.

Times in the format are microseconds, and torch writes them with three
decimals. The parser reads every number as a ``Decimal`` and converts
``ts`` and ``dur`` to whole integer nanoseconds, so the analysis and its
verifier add exact integers and never depend on float rounding. A time that
is not a whole nanosecond is refused.

Reads reuse the bounded reader of ``memory_access_report_io``: regular
non-symlink files only, with a byte bound. Duplicate keys, non-finite
constants, unknown fields and wrong field types all raise a subclass of
``ProfileTraceError``. Nothing here imports torch, so CI checks the parser
with recorded traces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .memory_access_pattern import MemoryAccessError
from .memory_access_report_io import read_bounded_bytes

MAX_TRACE_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 1 << 18
MAX_NAME_CHARS = 512
# 2**62 ns is about 146 years, far past any monotonic clock torch reads.
MAX_TIME_NS = 1 << 62
MAX_INTEGER_ID = (1 << 63) - 1
MAX_DEPTH = 256
NANOSECONDS_PER_MICROSECOND = 1000

COMPLETE = "X"
BEGIN = "B"
END = "E"
INSTANT = "i"
METADATA = "M"
PHASES = (COMPLETE, BEGIN, END, INSTANT, METADATA)
INSTANT_SCOPES = ("g", "p", "t")
# The metadata names of the format, each with the one argument it carries.
METADATA_ARGUMENTS = {
    "process_name": ("name", str),
    "process_labels": ("labels", str),
    "process_sort_index": ("sort_index", int),
    "thread_name": ("name", str),
    "thread_sort_index": ("sort_index", int),
}

_COMMON = {"ph", "pid", "tid"}
_REQUIRED = {
    COMPLETE: _COMMON | {"name", "ts", "dur"},
    BEGIN: _COMMON | {"name", "ts"},
    END: _COMMON | {"ts"},
    INSTANT: _COMMON | {"name", "ts"},
    METADATA: _COMMON | {"name", "args"},
}
_OPTIONAL = {
    COMPLETE: {"cat", "args"},
    BEGIN: {"cat", "args"},
    END: {"name", "cat", "args"},
    INSTANT: {"cat", "args", "s"},
    METADATA: {"cat", "ts"},
}
# Top-level fields that torch 2.15 writes beside traceEvents, with their types.
_TOP_LEVEL = {
    "schemaVersion": int,
    "deviceProperties": list,
    "trace_id": str,
    "record_shapes": int,
    "host_name": str,
    "displayTimeUnit": str,
    "baseTimeNanoseconds": int,
    "traceName": str,
}
DISPLAY_TIME_UNITS = ("ms", "ns")


class ProfileTraceError(ValueError):
    """Raised when a profiler trace, its analysis or its report is invalid."""


class TraceSizeError(ProfileTraceError):
    """Raised when a trace file or its event list is larger than the bounds."""


class TraceSyntaxError(ProfileTraceError):
    """Raised when a trace is not strict, duplicate-free UTF-8 JSON."""


class TraceFieldError(ProfileTraceError):
    """Raised when an event or the root has a missing, unknown or mistyped field."""


class TracePhaseError(ProfileTraceError):
    """Raised when an event uses a phase that this parser does not accept."""


class TraceTimeError(ProfileTraceError):
    """Raised when a time is negative, too large or not a whole nanosecond."""


class TraceNestingError(ProfileTraceError):
    """Raised when spans on one thread overlap without nesting, or B and E differ."""


def require_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return a true integer inside ``[minimum, maximum]``, never a JSON bool."""

    if type(value) is not int or not minimum <= value <= maximum:
        raise ProfileTraceError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TraceSyntaxError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise TraceSyntaxError(f"non-finite JSON constant: {name}")


def decode_trace_json(payload: bytes) -> dict[str, object]:
    """Decode strict UTF-8 JSON with unique keys and exact ``Decimal`` numbers."""

    if type(payload) is not bytes:
        raise TypeError("trace payload must be bytes")
    if len(payload) > MAX_TRACE_BYTES:
        raise TraceSizeError(f"trace exceeds {MAX_TRACE_BYTES} bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=Decimal,
        )
    except TraceSyntaxError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise TraceSyntaxError("trace is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TraceFieldError("trace root must be a JSON object")
    return value


def _name(value: object, label: str) -> str:
    if type(value) is not str or len(value) > MAX_NAME_CHARS:
        raise TraceFieldError(f"{label} must be a string of at most {MAX_NAME_CHARS}")
    return value


def _identifier(value: object, label: str) -> int | str:
    """Accept an integer id or a string id, as torch writes for its own tracks."""

    if type(value) is int:
        if not 0 <= value <= MAX_INTEGER_ID:
            raise TraceFieldError(f"{label} must be a non-negative 64-bit integer")
        return value
    if type(value) is str:
        return _name(value, label)
    raise TraceFieldError(f"{label} must be an integer or a string")


def microseconds_to_ns(value: object, label: str) -> int:
    """Convert a JSON microsecond value to exact integer nanoseconds."""

    if type(value) is int:
        nanoseconds = value * NANOSECONDS_PER_MICROSECOND
    elif type(value) is Decimal:
        nanoseconds = _decimal_to_ns(value, label)
    else:
        raise TraceFieldError(f"{label} must be a JSON number")
    if not 0 <= nanoseconds <= MAX_TIME_NS:
        raise TraceTimeError(f"{label} must be from 0 to {MAX_TIME_NS} ns")
    return nanoseconds


def _decimal_to_ns(exact: Decimal, label: str) -> int:
    """Scale a decimal by 1000 on its digits, never in a rounding context.

    Decimal arithmetic rounds to 28 digits and can overflow, and ``int`` of
    a huge exponent takes seconds. The digit tuple avoids all three, and the
    range check runs before any digit is joined.
    """

    if not exact.is_finite():
        raise TraceTimeError(f"{label} is not a finite number")
    sign, digits, exponent = exact.as_tuple()
    if not any(digits):
        return 0
    if sign or exact.adjusted() + 3 > len(str(MAX_TIME_NS)):
        raise TraceTimeError(f"{label} must be from 0 to {MAX_TIME_NS} ns")
    shift = exponent + 3
    if shift < 0:
        if len(digits) < -shift or any(digits[shift:]):
            raise TraceTimeError(f"{label} must be a whole number of nanoseconds")
        digits = digits[:shift]
        shift = 0
    return int("".join(map(str, digits)) or "0") * 10**shift


@dataclass(frozen=True)
class TraceSpan:
    """One timed region on one thread, from an ``X`` event or a ``B``/``E`` pair."""

    name: str
    category: str | None
    pid: int | str
    tid: int | str
    start_ns: int
    duration_ns: int

    def __post_init__(self) -> None:
        _name(self.name, "span name")
        if self.category is not None:
            _name(self.category, "span category")
        _identifier(self.pid, "span pid")
        _identifier(self.tid, "span tid")
        for value, label in ((self.start_ns, "start"), (self.duration_ns, "duration")):
            if type(value) is not int or not 0 <= value <= MAX_TIME_NS:
                raise TraceTimeError(f"span {label} must be from 0 to {MAX_TIME_NS} ns")
        if self.end_ns > MAX_TIME_NS:
            raise TraceTimeError(f"span end must be at most {MAX_TIME_NS} ns")

    @property
    def end_ns(self) -> int:
        """The first nanosecond after the span."""

        return self.start_ns + self.duration_ns

    @property
    def thread(self) -> tuple[int | str, int | str]:
        """The ``(pid, tid)`` track the span belongs to."""

        return self.pid, self.tid


@dataclass(frozen=True)
class TraceInstant:
    """One ``i`` event: a mark in time with a global, process or thread scope."""

    name: str
    category: str | None
    pid: int | str
    tid: int | str
    ts_ns: int
    scope: str


@dataclass(frozen=True)
class TraceMetadata:
    """One ``M`` event, reduced to the single argument its name defines."""

    name: str
    pid: int | str
    tid: int | str
    value: int | str


@dataclass(frozen=True)
class NestedSpan:
    """One span with its place in its thread's call tree.

    ``parent`` is the position in ``ParsedTrace.nested`` of the innermost
    span that contains this one on the same thread, or ``None`` for a
    top-level call.
    """

    span: TraceSpan
    parent: int | None
    depth: int


def nest_spans(spans: tuple[TraceSpan, ...]) -> tuple[NestedSpan, ...]:
    """Order spans by thread and start, and refuse any overlap that does not nest.

    A profiler records calls on one thread as a stack, so two spans on one
    thread either nest or are disjoint; a partial overlap means the trace
    was edited or merged wrongly, and self time would be undefined. Threads
    keep the order of their first span in the file. At one start, the
    longer span is the parent, and spans with equal positive bounds nest in
    file order. Spans are half-open, so a span that starts at another's end
    is disjoint from it, and a zero-length span contains nothing.
    """

    threads: dict[tuple[int | str, int | str], list[tuple[int, TraceSpan]]] = {}
    for position, span in enumerate(spans):
        if not isinstance(span, TraceSpan):
            raise TypeError("spans must be TraceSpan values")
        threads.setdefault(span.thread, []).append((position, span))
    nested: list[NestedSpan] = []
    for (pid, tid), members in threads.items():
        members.sort(key=lambda item: (item[1].start_ns, -item[1].duration_ns, item[0]))
        stack: list[int] = []
        for _, span in members:
            while stack and nested[stack[-1]].span.end_ns <= span.start_ns:
                stack.pop()
            if stack and span.end_ns > nested[stack[-1]].span.end_ns:
                outer = nested[stack[-1]].span
                raise TraceNestingError(
                    f"span {span.name!r} at {span.start_ns} ns overlaps "
                    f"{outer.name!r} on pid {pid!r} tid {tid!r} without nesting"
                )
            if len(stack) >= MAX_DEPTH:
                raise TraceNestingError(f"spans nest deeper than {MAX_DEPTH} levels")
            nested.append(
                NestedSpan(span, stack[-1] if stack else None, depth=len(stack))
            )
            stack.append(len(nested) - 1)
    return tuple(nested)


@dataclass(frozen=True)
class ParsedTrace:
    """The accepted content of one trace, in file order within each kind.

    ``nested`` holds the same spans as ``spans``, ordered by thread and
    start, each with its parent and depth.
    """

    spans: tuple[TraceSpan, ...]
    nested: tuple[NestedSpan, ...]
    instants: tuple[TraceInstant, ...]
    metadata: tuple[TraceMetadata, ...]
    event_count: int
    display_time_unit: str | None

    def thread_names(self) -> dict[tuple[int | str, int | str], str]:
        """Map each named ``(pid, tid)`` track to its last ``thread_name``."""

        return {
            (item.pid, item.tid): item.value
            for item in self.metadata
            if item.name == "thread_name"
        }


def _fields(event: dict[str, object], phase: str, index: int) -> None:
    keys = set(event)
    missing = _REQUIRED[phase] - keys
    unknown = keys - _REQUIRED[phase] - _OPTIONAL[phase]
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise TraceFieldError(
            f"event {index} ({phase}) fields do not match: {'; '.join(details)}"
        )
    if "args" in event and type(event["args"]) is not dict:
        raise TraceFieldError(f"event {index} args must be a JSON object")


def _category(event: dict[str, object], index: int) -> str | None:
    if "cat" not in event:
        return None
    return _name(event["cat"], f"event {index} cat")


def _metadata(event: dict[str, object], index: int, pid, tid) -> TraceMetadata:
    name = _name(event["name"], f"event {index} name")
    if name not in METADATA_ARGUMENTS:
        raise TraceFieldError(
            f"event {index} metadata name must be one of "
            f"{', '.join(METADATA_ARGUMENTS)}"
        )
    if "ts" in event:
        microseconds_to_ns(event["ts"], f"event {index} ts")
    if "cat" in event:
        _category(event, index)
    key, kind = METADATA_ARGUMENTS[name]
    args = event["args"]
    if set(args) != {key} or type(args[key]) is not kind:
        raise TraceFieldError(
            f"event {index} {name} args must hold only {key} as {kind.__name__}"
        )
    value = args[key]
    if kind is str:
        _name(value, f"event {index} {name}")
    elif not -MAX_INTEGER_ID <= value <= MAX_INTEGER_ID:
        raise TraceFieldError(f"event {index} {key} must fit 64 bits")
    return TraceMetadata(name, pid, tid, value)


def _close(
    stack: list[tuple[int, str, str | None]],
    event: dict[str, object],
    index: int,
    ts: int,
) -> TraceSpan:
    """Close the innermost open ``B`` event of this thread with one ``E`` event.

    The format pairs B and E by stack order on one thread, so an E closes the
    latest open B; a name on the E, when present, must repeat that B's name.
    """

    if not stack:
        raise TraceNestingError(f"E event {index} has no open B event on its thread")
    start, name, category = stack.pop()
    if "name" in event and _name(event["name"], f"event {index} name") != name:
        raise TraceNestingError(
            f"E event {index} closes {event['name']!r}, but the open B is {name!r}"
        )
    if ts < start:
        raise TraceNestingError(f"E event {index} ends before its B event starts")
    return TraceSpan(name, category, event["pid"], event["tid"], start, ts - start)


def _root(document: dict[str, object]) -> list[object]:
    unknown = set(document) - set(_TOP_LEVEL) - {"traceEvents"}
    if unknown:
        raise TraceFieldError(f"unknown trace fields: {', '.join(sorted(unknown))}")
    if "traceEvents" not in document:
        raise TraceFieldError("trace has no traceEvents array")
    for key, kind in _TOP_LEVEL.items():
        if key in document and type(document[key]) is not kind:
            raise TraceFieldError(f"trace field {key} must be {kind.__name__}")
    unit = document.get("displayTimeUnit")
    if unit is not None and unit not in DISPLAY_TIME_UNITS:
        raise TraceFieldError(
            f"displayTimeUnit must be one of {', '.join(DISPLAY_TIME_UNITS)}"
        )
    events = document["traceEvents"]
    if type(events) is not list:
        raise TraceFieldError("traceEvents must be a JSON array")
    if len(events) > MAX_EVENTS:
        raise TraceSizeError(f"trace holds more than {MAX_EVENTS} events")
    return events


def parse_trace(document: dict[str, object]) -> ParsedTrace:
    """Validate one decoded trace object and return its spans, instants and metadata."""

    if type(document) is not dict:
        raise TraceFieldError("trace root must be a JSON object")
    events = _root(document)
    spans: list[TraceSpan] = []
    open_spans: dict[
        tuple[int | str, int | str], list[tuple[int, str, str | None]]
    ] = {}
    instants: list[TraceInstant] = []
    metadata: list[TraceMetadata] = []
    for index, event in enumerate(events):
        if type(event) is not dict:
            raise TraceFieldError(f"event {index} must be a JSON object")
        phase = event.get("ph")
        if phase not in PHASES:
            raise TracePhaseError(
                f"event {index} phase {phase!r} is not one of {', '.join(PHASES)}"
            )
        _fields(event, phase, index)
        pid = _identifier(event["pid"], f"event {index} pid")
        tid = _identifier(event["tid"], f"event {index} tid")
        if phase == METADATA:
            metadata.append(_metadata(event, index, pid, tid))
            continue
        category = _category(event, index)
        ts = microseconds_to_ns(event["ts"], f"event {index} ts")
        if phase == COMPLETE:
            name = _name(event["name"], f"event {index} name")
            duration = microseconds_to_ns(event["dur"], f"event {index} dur")
            if ts + duration > MAX_TIME_NS:
                raise TraceTimeError(f"event {index} ends after {MAX_TIME_NS} ns")
            spans.append(TraceSpan(name, category, pid, tid, ts, duration))
        elif phase == INSTANT:
            scope = event.get("s", "t")
            if scope not in INSTANT_SCOPES:
                raise TraceFieldError(
                    f"event {index} scope must be one of {', '.join(INSTANT_SCOPES)}"
                )
            name = _name(event["name"], f"event {index} name")
            instants.append(TraceInstant(name, category, pid, tid, ts, scope))
        elif phase == BEGIN:
            name = _name(event["name"], f"event {index} name")
            open_spans.setdefault((pid, tid), []).append((ts, name, category))
        else:
            spans.append(_close(open_spans.get((pid, tid), []), event, index, ts))
    for (pid, tid), stack in open_spans.items():
        if stack:
            raise TraceNestingError(
                f"B event {stack[-1][1]!r} on pid {pid!r} tid {tid!r} has no E event"
            )
    return ParsedTrace(
        spans=tuple(spans),
        nested=nest_spans(tuple(spans)),
        instants=tuple(instants),
        metadata=tuple(metadata),
        event_count=len(events),
        display_time_unit=document.get("displayTimeUnit"),
    )


def parse_trace_bytes(payload: bytes) -> ParsedTrace:
    """Decode and validate one trace held in memory."""

    return parse_trace(decode_trace_json(payload))


def read_trace_bytes(path: Path) -> bytes:
    """Read one trace file with the shared bounded, non-symlink reader."""

    try:
        return read_bounded_bytes(path, max_bytes=MAX_TRACE_BYTES, label="trace")
    except MemoryAccessError as error:
        if "exceeds" in str(error):
            raise TraceSizeError(str(error)) from error
        raise ProfileTraceError(str(error)) from error


def load_trace(path: Path) -> ParsedTrace:
    """Read and validate one trace file."""

    return parse_trace_bytes(read_trace_bytes(path))
