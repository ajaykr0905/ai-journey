"""Independent verification of chained recovery traces."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .recovery_config import RecoveryExperimentError
from .recovery_trace import RecoveryEvent, chain_recovery_event

_RECORD_FIELDS = {
    "sequence",
    "run_id",
    "attempt_id",
    "kind",
    "monotonic_ns",
    "completed_items",
    "payload",
    "previous_sha256",
    "event_sha256",
}


def verify_recovery_trace(
    records: Sequence[dict[str, Any]],
) -> tuple[RecoveryEvent, ...]:
    """Verify exact records, chain ancestry, time, progress, and attempt changes."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence")
    if not records:
        raise RecoveryExperimentError("recovery trace must not be empty")
    events: list[RecoveryEvent] = []
    previous_digest = None
    run_identity = None
    attempt_identity = None
    monotonic_ns = -1
    completed_items = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise RecoveryExperimentError("recovery trace fields do not match schema")
        event = RecoveryEvent(
            **{
                key: value
                for key, value in record.items()
                if key not in {"previous_sha256", "event_sha256"}
            }
        )
        if event.sequence != index:
            raise RecoveryExperimentError("recovery trace sequence is not contiguous")
        if record["previous_sha256"] != previous_digest:
            raise RecoveryExperimentError("recovery trace ancestry changed")
        expected = chain_recovery_event(event, previous_digest)
        if not isinstance(record["event_sha256"], str) or record != expected:
            raise RecoveryExperimentError("recovery trace digest does not match")
        if event.monotonic_ns < monotonic_ns:
            raise RecoveryExperimentError("recovery trace time moved backwards")
        if index == 0:
            if event.kind != "started" or event.completed_items != 0:
                raise RecoveryExperimentError("recovery trace must begin with started")
            run_identity, attempt_identity = event.run_id, event.attempt_id
        else:
            if event.run_id != run_identity:
                raise RecoveryExperimentError("recovery trace run identity changed")
            if event.kind == "resumed":
                if event.attempt_id == attempt_identity:
                    raise RecoveryExperimentError(
                        "resumed event must start a new attempt"
                    )
                if event.completed_items > completed_items:
                    raise RecoveryExperimentError(
                        "resumed progress cannot exceed observed progress"
                    )
                attempt_identity = event.attempt_id
            elif event.attempt_id != attempt_identity:
                raise RecoveryExperimentError(
                    "attempt identity changed without a resumed event"
                )
            elif event.completed_items < completed_items:
                raise RecoveryExperimentError("completed_items moved backwards")
        previous_digest = record["event_sha256"]
        monotonic_ns = event.monotonic_ns
        completed_items = event.completed_items
        events.append(event)
    if events[-1].kind != "completed":
        raise RecoveryExperimentError("recovery trace must end with completed")
    json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return tuple(events)
