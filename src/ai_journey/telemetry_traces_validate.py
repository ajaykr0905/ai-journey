"""Strict structural validation of OTLP/JSON trace export requests.

The validator reads the JSON itself instead of trusting the span model that
produced it. It accepts the OTLP/JSON subset this lab emits, which is stricter
than what the OTLP specification asks of receivers: unknown fields are
rejected; trace IDs must be 32 and span IDs 16 nonzero lowercase hexadecimal
digits; times must be decimal strings of unsigned 64-bit nanoseconds with the
start no later than the end; span kinds and status codes must be known
integers; attribute values must be strings, booleans or decimal-string 64-bit
integers under unique semantic-convention keys.

Across the request, span IDs are unique within their trace, a trace lives in
one scope of one resource and has exactly one root, every ``parentSpanId``
names an earlier span of the same trace whose interval contains the child,
spans in a scope are ordered by start time, and events lie in order inside
their span. Every resource declares ``service.name``, the local CPU lab
environment label and an evidence boundary statement.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .telemetry_traces_io import TelemetryVerificationError
from .telemetry_traces_otlp import (
    MAX_ATTRIBUTES,
    MAX_EVENTS,
    MAX_SPANS,
    MAX_STRING_VALUE,
    MAX_UNIX_NANO,
    SPAN_KINDS,
)

ENVIRONMENT_ATTRIBUTE = "deployment.environment.name"
LOCAL_CPU_LAB = "local-cpu-lab"
BOUNDARY_ATTRIBUTE = "ai_journey.evidence.boundary"
MAX_RESOURCES = 64
_KIND_NAMES = {number: name for name, number in SPAN_KINDS.items()}
_HEX = re.compile(r"[0-9a-f]+")
_UNSIGNED = re.compile(r"0|[1-9][0-9]{0,19}")
_SIGNED = re.compile(r"-?(?:0|[1-9][0-9]{0,18})")
_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
_NAME = re.compile(r"[!-~](?:[ -~]{0,126}[!-~])?")
_TEXT = re.compile(r"[ -~]*")
_SPAN_FIELDS = frozenset(
    {
        "traceId",
        "spanId",
        "name",
        "kind",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "attributes",
        "status",
    }
)


class OtlpValidationError(TelemetryVerificationError):
    """Raised when an OTLP/JSON trace export breaks the structural contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OtlpValidationError(message)


def _object(
    value: object, required: frozenset[str], optional: frozenset[str], where: str
) -> dict[str, Any]:
    _require(type(value) is dict, f"{where} must be a JSON object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    _require(not missing, f"{where} lacks {', '.join(missing)}")
    _require(not unknown, f"{where} has unknown field {', '.join(unknown)}")
    return value


def _list(value: object, where: str, *, limit: int) -> list[Any]:
    _require(
        type(value) is list and 1 <= len(value) <= limit,
        f"{where} must be a list of 1 to {limit} items",
    )
    return value


def _identifier(value: object, length: int, where: str) -> str:
    _require(
        type(value) is str
        and len(value) == length
        and _HEX.fullmatch(value) is not None
        and int(value, 16) != 0,
        f"{where} must be {length} nonzero lowercase hex digits",
    )
    return value


def _nanoseconds(value: object, where: str) -> int:
    _require(
        type(value) is str and _UNSIGNED.fullmatch(value) is not None,
        f"{where} must be a decimal string of nanoseconds",
    )
    number = int(value)
    _require(
        1 <= number <= MAX_UNIX_NANO, f"{where} must fit unsigned 64-bit nanoseconds"
    )
    return number


def _name(value: object, where: str) -> str:
    _require(
        type(value) is str and _NAME.fullmatch(value) is not None,
        f"{where} must be 1 to 128 printable ASCII characters",
    )
    return value


def _any_value(value: object, where: str) -> str | bool | int:
    _require(
        type(value) is dict and len(value) == 1,
        f"{where} must hold exactly one typed value",
    )
    ((kind, item),) = value.items()
    if kind == "stringValue":
        _require(
            type(item) is str
            and len(item) <= MAX_STRING_VALUE
            and _TEXT.fullmatch(item) is not None,
            f"{where} must be a short printable ASCII string",
        )
        return item
    if kind == "boolValue":
        _require(type(item) is bool, f"{where} must be a JSON boolean")
        return item
    _require(kind == "intValue", f"{where} uses unsupported value type {kind}")
    _require(
        type(item) is str
        and _SIGNED.fullmatch(item) is not None
        and -(2**63) <= int(item) <= 2**63 - 1,
        f"{where} must be a decimal string of a 64-bit integer",
    )
    return int(item)


def _attributes(value: object, where: str) -> dict[str, str | bool | int]:
    _require(
        type(value) is list and len(value) <= MAX_ATTRIBUTES,
        f"{where} must be a list of at most {MAX_ATTRIBUTES} attributes",
    )
    attributes: dict[str, str | bool | int] = {}
    for index, item in enumerate(value):
        entry = _object(
            item, frozenset({"key", "value"}), frozenset(), f"{where}[{index}]"
        )
        key = entry["key"]
        _require(
            type(key) is str and _KEY.fullmatch(key) is not None,
            f"{where}[{index}] key is not a lowercase dotted name",
        )
        _require(key not in attributes, f"{where} repeats key {key}")
        attributes[key] = _any_value(entry["value"], f"{where}.{key}")
    return attributes


@dataclass(frozen=True)
class ExportSummary:
    """Counts that a sealed report binds to one validated export request."""

    resources: int
    trace_ids: tuple[str, ...]
    spans: int
    events: int
    spans_by_kind: dict[str, int]
    error_spans: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "resources": self.resources,
            "trace_ids": list(self.trace_ids),
            "spans": self.spans,
            "events": self.events,
            "spans_by_kind": dict(self.spans_by_kind),
            "error_spans": self.error_spans,
        }


class _Spans:
    """Accumulate request-wide span identity, parent and ordering checks."""

    def __init__(self) -> None:
        self.intervals: dict[tuple[str, str], tuple[int, int]] = {}
        self.scopes: dict[str, tuple[int, int]] = {}
        self.roots: set[str] = set()
        self.kinds: Counter[str] = Counter()
        self.events = 0
        self.errors = 0

    def add(self, span: object, scope: tuple[int, int], where: str) -> int:
        entry = _object(
            span, _SPAN_FIELDS, frozenset({"parentSpanId", "events"}), where
        )
        trace_id = _identifier(entry["traceId"], 32, f"{where}.traceId")
        span_id = _identifier(entry["spanId"], 16, f"{where}.spanId")
        _name(entry["name"], f"{where}.name")
        kind = entry["kind"]
        _require(
            type(kind) is int and kind in _KIND_NAMES,
            f"{where}.kind must be a known SpanKind integer",
        )
        start = _nanoseconds(entry["startTimeUnixNano"], f"{where}.startTimeUnixNano")
        end = _nanoseconds(entry["endTimeUnixNano"], f"{where}.endTimeUnixNano")
        _require(start <= end, f"{where} ends before it starts")
        _attributes(entry["attributes"], f"{where}.attributes")
        _require(
            self.scopes.setdefault(trace_id, scope) == scope,
            f"{where} splits trace {trace_id} across scopes",
        )
        key = (trace_id, span_id)
        _require(key not in self.intervals, f"{where} repeats span {span_id}")
        if "parentSpanId" in entry:
            parent_id = _identifier(entry["parentSpanId"], 16, f"{where}.parentSpanId")
            _require(parent_id != span_id, f"{where} is its own parent")
            parent = self.intervals.get((trace_id, parent_id))
            _require(
                parent is not None,
                f"{where}.parentSpanId names no earlier span of its trace",
            )
            _require(
                parent[0] <= start and end <= parent[1],
                f"{where} lies outside its parent span",
            )
        else:
            _require(trace_id not in self.roots, f"{where} is a second root")
            self.roots.add(trace_id)
        self.intervals[key] = (start, end)
        self._events(entry.get("events"), start, end, where)
        self._status(entry["status"], where)
        self.kinds[_KIND_NAMES[kind]] += 1
        return start

    def _events(self, events: object, start: int, end: int, where: str) -> None:
        if events is None:
            return
        previous = start
        for index, event in enumerate(
            _list(events, f"{where}.events", limit=MAX_EVENTS)
        ):
            label = f"{where}.events[{index}]"
            entry = _object(
                event,
                frozenset({"timeUnixNano", "name"}),
                frozenset({"attributes"}),
                label,
            )
            time = _nanoseconds(entry["timeUnixNano"], f"{label}.timeUnixNano")
            _require(
                previous <= time <= end, f"{label} is out of order or outside its span"
            )
            _name(entry["name"], f"{label}.name")
            _attributes(entry.get("attributes", []), f"{label}.attributes")
            previous = time
            self.events += 1

    def _status(self, status: object, where: str) -> None:
        entry = _object(
            status, frozenset({"code"}), frozenset({"message"}), f"{where}.status"
        )
        code = entry["code"]
        _require(
            type(code) is int and code in (0, 1, 2),
            f"{where}.status.code must be 0, 1 or 2",
        )
        if "message" in entry:
            _require(code == 2, f"{where}.status carries a message without an error")
            _name(entry["message"], f"{where}.status.message")
        if code == 2:
            self.errors += 1


def validate_export_request(payload: object) -> ExportSummary:
    """Check one ExportTraceServiceRequest object and summarize what it holds."""

    request = _object(payload, frozenset({"resourceSpans"}), frozenset(), "request")
    spans = _Spans()
    resources = _list(request["resourceSpans"], "resourceSpans", limit=MAX_RESOURCES)
    total = 0
    for r_index, item in enumerate(resources):
        where = f"resourceSpans[{r_index}]"
        entry = _object(item, frozenset({"resource", "scopeSpans"}), frozenset(), where)
        resource = _object(
            entry["resource"],
            frozenset({"attributes"}),
            frozenset(),
            f"{where}.resource",
        )
        attributes = _attributes(resource["attributes"], f"{where}.resource.attributes")
        _require(
            type(attributes.get("service.name")) is str
            and bool(attributes["service.name"]),
            f"{where}.resource needs a service.name",
        )
        _require(
            attributes.get(ENVIRONMENT_ATTRIBUTE) == LOCAL_CPU_LAB,
            f"{where}.resource must be labeled {ENVIRONMENT_ATTRIBUTE}={LOCAL_CPU_LAB}",
        )
        _require(
            type(attributes.get(BOUNDARY_ATTRIBUTE)) is str
            and bool(attributes[BOUNDARY_ATTRIBUTE]),
            f"{where}.resource needs an {BOUNDARY_ATTRIBUTE} statement",
        )
        scopes = _list(entry["scopeSpans"], f"{where}.scopeSpans", limit=MAX_RESOURCES)
        for s_index, scope_spans in enumerate(scopes):
            label = f"{where}.scopeSpans[{s_index}]"
            scope_entry = _object(
                scope_spans, frozenset({"scope", "spans"}), frozenset(), label
            )
            scope = _object(
                scope_entry["scope"],
                frozenset({"name"}),
                frozenset({"version"}),
                f"{label}.scope",
            )
            _name(scope["name"], f"{label}.scope.name")
            if "version" in scope:
                _name(scope["version"], f"{label}.scope.version")
            items = _list(scope_entry["spans"], f"{label}.spans", limit=MAX_SPANS)
            total += len(items)
            _require(total <= MAX_SPANS, f"request exceeds {MAX_SPANS} spans")
            previous = 0
            for p_index, span in enumerate(items):
                start = spans.add(span, (r_index, s_index), f"{label}.spans[{p_index}]")
                _require(
                    start >= previous, f"{label}.spans are not ordered by start time"
                )
                previous = start
    return ExportSummary(
        resources=len(resources),
        trace_ids=tuple(sorted(spans.roots)),
        spans=total,
        events=spans.events,
        spans_by_kind=dict(sorted(spans.kinds.items())),
        error_spans=spans.errors,
    )
