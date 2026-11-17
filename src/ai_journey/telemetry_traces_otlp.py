"""OpenTelemetry span model and OTLP/JSON ExportTraceServiceRequest encoding.

The labs time everything with ``time.monotonic_ns`` in the experiment process,
whose epoch is arbitrary and which OTLP cannot carry. ``MonotonicEpochMapping``
states the conversion instead of hiding it::

    unix_nano = anchor_unix_nano + (monotonic_ns - origin_monotonic_ns)

The origin is the run's first lifecycle event and the anchor is a declared
instant, so every duration and ordering is an exact measured difference while
the absolute timestamps are synthetic.

Encoding follows the OTLP JSON mapping of ``ExportTraceServiceRequest`` from
opentelemetry-proto ``trace/v1``: lowerCamelCase field names, trace and span
IDs as hexadecimal strings rather than base64, enums as integers, and 64-bit
integers (times and ``intValue``) as decimal strings. Attribute values are
limited to strings, booleans and 64-bit integers. Spans are ordered by start
time with each parent before its children and attributes by key, so the bytes
are a pure function of the spans.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .telemetry_traces_io import TelemetryError
from .trace_context import TraceContext

SPAN_KINDS = {"internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5}
STATUS_CODES = {"unset": 0, "ok": 1, "error": 2}
MONOTONIC_CLOCK = "time.monotonic_ns in the experiment process on one host"
TIME_MAPPING_FORMULA = (
    "unix_nano = anchor_unix_nano + (monotonic_ns - origin_monotonic_ns)"
)
MAX_UNIX_NANO = 2**64 - 1
MAX_MONOTONIC_NS = 2**63 - 1
# The OpenTelemetry SDK's default per-span attribute and event count limits.
MAX_ATTRIBUTES = 128
MAX_EVENTS = 128
MAX_SPANS = 20_000
MAX_STRING_VALUE = 4096
_ATTRIBUTE_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")
_NAME = re.compile(r"[!-~](?:[ -~]{0,126}[!-~])?")
_TEXT = re.compile(r"[ -~]*")

AttributeValue = str | bool | int


def _require_time(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_UNIX_NANO:
        raise TelemetryError(f"{name} must be an integer in [1, 2**64 - 1]")
    return value


def _require_name(value: object, name: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise TelemetryError(f"{name} must be 1 to 128 printable ASCII characters")
    return value


def require_attributes(attributes: object, name: str) -> dict[str, AttributeValue]:
    """Copy attributes after checking semantic-convention key syntax and types.

    Keys are lowercase dot-separated namespaces of snake_case words, the
    OpenTelemetry attribute naming rule. Values are printable ASCII strings of
    at most 4096 characters, booleans or 64-bit signed integers.
    """

    if not isinstance(attributes, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if len(attributes) > MAX_ATTRIBUTES:
        raise TelemetryError(f"{name} exceed {MAX_ATTRIBUTES} attributes")
    copied: dict[str, AttributeValue] = {}
    for key, value in attributes.items():
        if type(key) is not str or _ATTRIBUTE_KEY.fullmatch(key) is None:
            raise TelemetryError(f"{name} key {key!r} is not a lowercase dotted name")
        if type(value) is str:
            if len(value) > MAX_STRING_VALUE or _TEXT.fullmatch(value) is None:
                raise TelemetryError(f"{name} {key} must be short printable ASCII")
        elif type(value) is int:
            if not -(2**63) <= value <= 2**63 - 1:
                raise TelemetryError(f"{name} {key} overflows a 64-bit integer")
        elif type(value) is not bool:
            raise TelemetryError(f"{name} {key} must be a string, boolean or integer")
        copied[key] = value
    return copied


@dataclass(frozen=True)
class MonotonicEpochMapping:
    """Explicit affine map from one host's monotonic clock to Unix nanoseconds."""

    origin_monotonic_ns: int
    anchor_unix_nano: int

    def __post_init__(self) -> None:
        if (
            type(self.origin_monotonic_ns) is not int
            or not 0 <= self.origin_monotonic_ns <= MAX_MONOTONIC_NS
        ):
            raise TelemetryError("origin_monotonic_ns must be in [0, 2**63 - 1]")
        _require_time(self.anchor_unix_nano, "anchor_unix_nano")

    def to_unix_nano(self, monotonic_ns: int) -> int:
        """Map one monotonic reading taken at or after the origin."""

        if type(monotonic_ns) is not int:
            raise TelemetryError("monotonic_ns must be an integer")
        if monotonic_ns < self.origin_monotonic_ns:
            raise TelemetryError("monotonic_ns precedes the mapping origin")
        value = self.anchor_unix_nano + monotonic_ns - self.origin_monotonic_ns
        if value > MAX_UNIX_NANO:
            raise TelemetryError("mapped time overflows unsigned 64-bit nanoseconds")
        return value

    def to_mapping(self) -> dict[str, Any]:
        return {
            "clock": MONOTONIC_CLOCK,
            "formula": TIME_MAPPING_FORMULA,
            "origin_monotonic_ns": self.origin_monotonic_ns,
            "anchor_unix_nano": self.anchor_unix_nano,
        }


@dataclass(frozen=True)
class SpanEvent:
    """One timestamped point inside a span, such as an injected fault."""

    name: str
    time_unix_nano: int
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name(self.name, "event name")
        _require_time(self.time_unix_nano, "event time")
        object.__setattr__(
            self, "attributes", require_attributes(self.attributes, "event attributes")
        )


@dataclass(frozen=True)
class TelemetrySpan:
    """One finished span whose identity is a validated W3C trace context."""

    context: TraceContext
    name: str
    kind: str
    start_unix_nano: int
    end_unix_nano: int
    attributes: Mapping[str, AttributeValue]
    events: tuple[SpanEvent, ...] = ()
    status: str = "unset"
    status_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, TraceContext):
            raise TypeError("context must be TraceContext")
        _require_name(self.name, "span name")
        if self.kind not in SPAN_KINDS:
            raise TelemetryError(f"span kind must be one of {', '.join(SPAN_KINDS)}")
        _require_time(self.start_unix_nano, "span start")
        _require_time(self.end_unix_nano, "span end")
        if self.end_unix_nano < self.start_unix_nano:
            raise TelemetryError(f"span {self.name} ends before it starts")
        object.__setattr__(
            self, "attributes", require_attributes(self.attributes, "span attributes")
        )
        if type(self.events) is not tuple or any(
            not isinstance(event, SpanEvent) for event in self.events
        ):
            raise TypeError("events must be a tuple of SpanEvent values")
        if len(self.events) > MAX_EVENTS:
            raise TelemetryError(f"span {self.name} exceeds {MAX_EVENTS} events")
        times = [event.time_unix_nano for event in self.events]
        if times != sorted(times) or any(
            not self.start_unix_nano <= time <= self.end_unix_nano for time in times
        ):
            raise TelemetryError(f"span {self.name} events must be ordered inside it")
        if self.status not in STATUS_CODES:
            raise TelemetryError("span status must be unset, ok or error")
        # OpenTelemetry allows a status description only on errors.
        if self.status_message is not None and (
            self.status != "error"
            or type(self.status_message) is not str
            or _NAME.fullmatch(self.status_message) is None
        ):
            raise TelemetryError("only an error status carries a short message")

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def parent_span_id(self) -> str | None:
        return self.context.parent_span_id


def _span_depths(spans: tuple[TelemetrySpan, ...]) -> dict[tuple[str, str], int]:
    """Return each span's depth below its root; reject broken parent links."""

    by_id: dict[tuple[str, str], TelemetrySpan] = {}
    for span in spans:
        key = (span.trace_id, span.span_id)
        if key in by_id:
            raise TelemetryError("span IDs must be unique within a trace")
        by_id[key] = span
    depths: dict[tuple[str, str], int] = {}
    for span in spans:
        chain: list[tuple[str, str]] = []
        on_chain: set[tuple[str, str]] = set()
        key = (span.trace_id, span.span_id)
        while key not in depths:
            current = by_id[key]
            if current.parent_span_id is None:
                depths[key] = 0
                break
            chain.append(key)
            on_chain.add(key)
            key = (current.trace_id, current.parent_span_id)
            if key not in by_id:
                raise TelemetryError(f"span {current.name} names a missing parent")
            if key in on_chain:
                raise TelemetryError("span parent links form a cycle")
        for offset, item in enumerate(reversed(chain), start=1):
            depths[item] = depths[key] + offset
    return depths


@dataclass(frozen=True)
class ResourceTelemetry:
    """The complete traces of one lab run with the resource that produced them.

    Every trace has exactly one root, every parent is present and every child
    lies inside its parent's interval, so the encoded request is self-contained.
    """

    attributes: Mapping[str, AttributeValue]
    scope_name: str
    scope_version: str
    spans: tuple[TelemetrySpan, ...]

    def __post_init__(self) -> None:
        attributes = require_attributes(self.attributes, "resource attributes")
        if type(attributes.get("service.name")) is not str:
            raise TelemetryError("a resource needs a service.name string")
        object.__setattr__(self, "attributes", attributes)
        _require_name(self.scope_name, "scope name")
        _require_name(self.scope_version, "scope version")
        if type(self.spans) is not tuple or any(
            not isinstance(span, TelemetrySpan) for span in self.spans
        ):
            raise TypeError("spans must be a tuple of TelemetrySpan values")
        if not 1 <= len(self.spans) <= MAX_SPANS:
            raise TelemetryError(f"a resource holds 1 to {MAX_SPANS} spans")
        depths = _span_depths(self.spans)
        by_id = {(span.trace_id, span.span_id): span for span in self.spans}
        roots: dict[str, int] = {}
        for span in self.spans:
            if depths[(span.trace_id, span.span_id)] == 0:
                roots[span.trace_id] = roots.get(span.trace_id, 0) + 1
                continue
            parent = by_id[(span.trace_id, span.parent_span_id)]
            if not (
                parent.start_unix_nano <= span.start_unix_nano
                and span.end_unix_nano <= parent.end_unix_nano
            ):
                raise TelemetryError(f"span {span.name} lies outside its parent")
        if any(count != 1 for count in roots.values()):
            raise TelemetryError("every trace needs exactly one root span")

    @property
    def trace_ids(self) -> frozenset[str]:
        return frozenset(span.trace_id for span in self.spans)

    def ordered_spans(self) -> tuple[TelemetrySpan, ...]:
        """Order by start time, then depth, so every parent precedes its children."""

        depths = _span_depths(self.spans)
        return tuple(
            sorted(
                self.spans,
                key=lambda span: (
                    span.start_unix_nano,
                    depths[(span.trace_id, span.span_id)],
                    span.trace_id,
                    span.span_id,
                ),
            )
        )


def encode_any_value(value: AttributeValue) -> dict[str, Any]:
    """Encode one attribute value as an OTLP JSON AnyValue."""

    if type(value) is bool:
        return {"boolValue": value}
    if type(value) is int:
        return {"intValue": str(value)}
    if type(value) is str:
        return {"stringValue": value}
    raise TelemetryError("attribute values must be strings, booleans or integers")


def _key_values(attributes: Mapping[str, AttributeValue]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": encode_any_value(attributes[key])}
        for key in sorted(attributes)
    ]


def encode_span(span: TelemetrySpan) -> dict[str, Any]:
    """Encode one span; roots omit ``parentSpanId``, eventless spans ``events``."""

    if not isinstance(span, TelemetrySpan):
        raise TypeError("span must be TelemetrySpan")
    encoded: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": SPAN_KINDS[span.kind],
        "startTimeUnixNano": str(span.start_unix_nano),
        "endTimeUnixNano": str(span.end_unix_nano),
        "attributes": _key_values(span.attributes),
        "status": {"code": STATUS_CODES[span.status]},
    }
    if span.parent_span_id is not None:
        encoded["parentSpanId"] = span.parent_span_id
    if span.status_message is not None:
        encoded["status"]["message"] = span.status_message
    if span.events:
        encoded["events"] = [
            {
                "timeUnixNano": str(event.time_unix_nano),
                "name": event.name,
                "attributes": _key_values(event.attributes),
            }
            for event in span.events
        ]
    return encoded


def encode_export_request(resources: Sequence[ResourceTelemetry]) -> dict[str, Any]:
    """Return one ExportTraceServiceRequest object with a ResourceSpans per run."""

    if not isinstance(resources, Sequence) or any(
        not isinstance(resource, ResourceTelemetry) for resource in resources
    ):
        raise TypeError("resources must be a sequence of ResourceTelemetry values")
    if not resources:
        raise TelemetryError("an export request needs at least one resource")
    seen: set[str] = set()
    for resource in resources:
        if seen & resource.trace_ids:
            raise TelemetryError("a trace must not be split across resources")
        seen |= resource.trace_ids
    if sum(len(resource.spans) for resource in resources) > MAX_SPANS:
        raise TelemetryError(f"an export request holds at most {MAX_SPANS} spans")
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _key_values(resource.attributes)},
                "scopeSpans": [
                    {
                        "scope": {
                            "name": resource.scope_name,
                            "version": resource.scope_version,
                        },
                        "spans": [encode_span(s) for s in resource.ordered_spans()],
                    }
                ],
            }
            for resource in resources
        ]
    }
