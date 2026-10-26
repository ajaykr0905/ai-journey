"""W3C traceparent parsing and child IDs without baggage or vendor state.

Only the public identifier fields from https://www.w3.org/TR/trace-context/
are retained. This is a propagation helper, not a tracing SDK or exporter.
"""

from dataclasses import dataclass
import re
import secrets


_TRACEPARENT = re.compile(
    r"([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(.*)"
)


def _identifier(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
        and int(value, 16) != 0
    )


@dataclass(frozen=True)
class TraceContext:
    """One local span identity with an optional known parent span."""

    trace_id: str
    span_id: str
    sampled: bool = False
    parent_span_id: str | None = None

    def __post_init__(self):
        if not _identifier(self.trace_id, 32) or not _identifier(self.span_id, 16):
            raise ValueError("trace and span IDs must be nonzero lowercase hexadecimal")
        if type(self.sampled) is not bool:
            raise ValueError("sampled must be boolean")
        if self.parent_span_id is not None and (
            not _identifier(self.parent_span_id, 16)
            or self.parent_span_id == self.span_id
        ):
            raise ValueError("parent span must be a distinct nonzero span ID")

    def to_header(self) -> str:
        """Emit known version-zero fields without unknown flags or extensions."""
        return f"00-{self.trace_id}-{self.span_id}-{int(self.sampled):02x}"


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Ignore malformed input and retain valid future-version base fields.

    HTTP optional whitespace is allowed around the value. Header values over
    512 characters, control characters, duplicate comma-joined values, all-zero
    IDs and forbidden version ff restart the trace instead of being propagated.
    """
    if not isinstance(value, str) or len(value) > 512:
        return None
    value = value.strip(" \t")
    if any(not 32 <= ord(character) <= 126 for character in value) or "," in value:
        return None
    match = _TRACEPARENT.fullmatch(value)
    if match is None:
        return None
    version, trace_id, span_id, flags, extension = match.groups()
    if version == "ff" or version == "00" and extension:
        return None
    if extension and not extension.startswith("-"):
        return None
    if not _identifier(trace_id, 32) or not _identifier(span_id, 16):
        return None
    return TraceContext(trace_id, span_id, bool(int(flags, 16) & 1))


def _random_identifier(byte_count, *, different_from=None):
    for _ in range(4):
        value = secrets.token_hex(byte_count)
        if _identifier(value, byte_count * 2) and value != different_from:
            return value
    raise RuntimeError("could not generate a distinct nonzero trace identifier")


def child_context(
    parent: TraceContext | None = None, *, sampled: bool | None = None
) -> TraceContext:
    """Create a distinct child span, or a new unsampled trace when parent is absent."""
    if parent is not None and not isinstance(parent, TraceContext):
        raise TypeError("parent must be TraceContext or None")
    if sampled is not None and type(sampled) is not bool:
        raise ValueError("sampled must be boolean or None")
    trace_id = parent.trace_id if parent is not None else _random_identifier(16)
    parent_span = parent.span_id if parent is not None else None
    span_id = _random_identifier(8, different_from=parent_span)
    decision = (
        (parent.sampled if parent is not None else False)
        if sampled is None
        else sampled
    )
    return TraceContext(trace_id, span_id, decision, parent_span)
