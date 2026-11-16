"""Deterministic W3C trace-context identifiers derived from lab run identities.

The recovery labs derive their run, lifetime and attempt identities from sealed
inputs, so their telemetry identifiers are derived as well. A trace ID is the
first 16 bytes of SHA-256 over the canonical JSON of the lab, its run identity
and the head digest of its verified hash-chained lifecycle trace; a span ID is
the first 8 bytes of SHA-256 over the trace ID and a stable span key such as
``lifetime/<lifetime_id>``. Exporting the same evidence twice therefore yields
the same IDs, while a second execution of one pinned config (same run ID, new
trace head) gets a different trace instead of colliding with the first.

Contexts are built and rendered through ``trace_context.TraceContext``, which
enforces nonzero lowercase hexadecimal IDs and a distinct parent. The IDs are
digests, not random draws, so the W3C Level 2 ``random`` trace flag is never
set; the sampled flag is set because every exported span was recorded.
"""

from __future__ import annotations

import re

from .telemetry_traces_io import TelemetryError, canonical_sha256
from .trace_context import TraceContext

LABS = ("serving-recovery", "worker-recovery")
TRACE_ID_KIND = "ai-journey-telemetry-trace"
SPAN_ID_KIND = "ai-journey-telemetry-span"
# A digest prefix of zero or equal to the parent is re-derived with a salt.
_DERIVATIONS = 4
_RUN_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SPAN_KEY = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}")


def _require_identifier(value: object, length: int, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or re.fullmatch(r"[0-9a-f]+", value) is None
        or int(value, 16) == 0
    ):
        raise TelemetryError(f"{name} must be {length} nonzero lowercase hex digits")
    return value


def _derive(kind: str, fields: dict[str, str], length: int, avoid: str | None) -> str:
    for salt in range(_DERIVATIONS):
        payload: dict[str, object] = {"kind": kind, **fields}
        if salt:
            payload["salt"] = salt
        value = canonical_sha256(payload)[:length]
        if int(value, 16) != 0 and value != avoid:
            return value
    raise TelemetryError("could not derive a distinct nonzero identifier")


def derive_trace_id(*, lab: str, run_id: str, trace_head_sha256: str) -> str:
    """Return the 32-hex trace ID of one verified run's lifecycle evidence."""

    if lab not in LABS:
        raise TelemetryError(f"lab must be one of {', '.join(LABS)}")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise TelemetryError("run_id is invalid")
    if (
        type(trace_head_sha256) is not str
        or _DIGEST.fullmatch(trace_head_sha256) is None
    ):
        raise TelemetryError("trace_head_sha256 must be a lowercase SHA-256 digest")
    fields = {"lab": lab, "run_id": run_id, "trace_head_sha256": trace_head_sha256}
    return _derive(TRACE_ID_KIND, fields, 32, None)


def derive_span_id(
    trace_id: str, span_key: str, *, parent_span_id: str | None = None
) -> str:
    """Return the 16-hex span ID of a stable span key, distinct from its parent."""

    _require_identifier(trace_id, 32, "trace_id")
    if parent_span_id is not None:
        _require_identifier(parent_span_id, 16, "parent_span_id")
    if type(span_key) is not str or _SPAN_KEY.fullmatch(span_key) is None:
        raise TelemetryError("span key must be a short lowercase path")
    fields = {"trace_id": trace_id, "span_key": span_key}
    return _derive(SPAN_ID_KIND, fields, 16, parent_span_id)


def span_context(
    trace_id: str, span_key: str, *, parent: TraceContext | None = None
) -> TraceContext:
    """Return the sampled context of one derived span under an optional parent."""

    if parent is not None:
        if not isinstance(parent, TraceContext):
            raise TypeError("parent must be TraceContext or None")
        if parent.trace_id != trace_id:
            raise TelemetryError("a parent span must belong to the same trace")
    parent_span_id = None if parent is None else parent.span_id
    span_id = derive_span_id(trace_id, span_key, parent_span_id=parent_span_id)
    return TraceContext(trace_id, span_id, True, parent_span_id)
