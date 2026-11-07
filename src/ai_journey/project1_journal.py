"""Chained lifecycle evidence for Project 1 training attempts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

from .project1_contract import Project1RunError

_ATTEMPT_ID = re.compile(r"attempt-[1-9][0-9]*-[0-9a-f]{16}")
_KINDS = {"started", "heartbeat", "step", "checkpoint", "failed", "completed"}


def _require_json_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Project1RunError(f"{name} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise Project1RunError(f"{name} keys must be non-empty strings")
            _require_json_value(item, f"{name}.{key}")
        return
    raise Project1RunError(f"{name} is not a canonical JSON value")


@dataclass(frozen=True)
class Project1Event:
    """One ordered, content-safe training lifecycle observation."""

    sequence: int
    attempt_id: str
    kind: str
    recorded_at_utc: str
    completed_step: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise Project1RunError("sequence must be non-negative")
        if (
            not isinstance(self.attempt_id, str)
            or _ATTEMPT_ID.fullmatch(self.attempt_id) is None
        ):
            raise Project1RunError("attempt_id is invalid")
        if self.kind not in _KINDS:
            raise Project1RunError("event kind is unsupported")
        try:
            timestamp = datetime.fromisoformat(self.recorded_at_utc)
        except (TypeError, ValueError) as exc:
            raise Project1RunError(
                "recorded_at_utc must be an ISO-8601 timestamp"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise Project1RunError("recorded_at_utc must use UTC")
        if isinstance(self.completed_step, bool) or not isinstance(
            self.completed_step, int
        ):
            raise TypeError("completed_step must be an integer")
        if self.completed_step < 0:
            raise Project1RunError("completed_step must be non-negative")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        _require_json_value(self.payload, "payload")
        json.dumps(self.payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of the event."""

        return {
            "sequence": self.sequence,
            "attempt_id": self.attempt_id,
            "kind": self.kind,
            "recorded_at_utc": self.recorded_at_utc,
            "completed_step": self.completed_step,
            "payload": dict(self.payload),
        }


def chain_event(
    event: Project1Event, previous_sha256: str | None = None
) -> dict[str, Any]:
    """Bind an event to every previously accepted journal record."""

    if not isinstance(event, Project1Event):
        raise TypeError("event must be Project1Event")
    if previous_sha256 is not None and (
        not isinstance(previous_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", previous_sha256) is None
    ):
        raise Project1RunError("previous_sha256 must be a SHA-256 digest")
    record = {"previous_sha256": previous_sha256, "event": event.to_mapping()}
    record["event_sha256"] = sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def verify_event_chain(records: object) -> tuple[Project1Event, ...]:
    """Verify sequence continuity, record digests, and hash-chain ancestry."""

    if not isinstance(records, list) or not records:
        raise Project1RunError("event chain must be a non-empty list")
    events: list[Project1Event] = []
    previous = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "previous_sha256",
            "event",
            "event_sha256",
        }:
            raise Project1RunError("event record fields do not match schema")
        if record["previous_sha256"] != previous:
            raise Project1RunError("event chain ancestry does not match")
        unsigned = {
            "previous_sha256": record["previous_sha256"],
            "event": record["event"],
        }
        expected = sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if record["event_sha256"] != expected:
            raise Project1RunError("event record digest does not match")
        try:
            event = Project1Event(**record["event"])
        except (TypeError, ValueError) as exc:
            raise Project1RunError("event record is invalid") from exc
        if event.sequence != index:
            raise Project1RunError("event sequence is not contiguous")
        events.append(event)
        previous = record["event_sha256"]
    return tuple(events)
