"""Canonical monotonic event evidence for worker recovery experiments."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .recovery_config import RecoveryExperimentError

_RUN_ID = re.compile(r"[a-z0-9-]+-[0-9a-f]{16}")
_ATTEMPT_ID = re.compile(r"attempt-[1-9][0-9]*-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_KINDS = {
    "started",
    "item_completed",
    "heartbeat",
    "checkpoint",
    "fault_injected",
    "failure_detected",
    "resumed",
    "completed",
}


def _require_json_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise RecoveryExperimentError(f"{name} keys must be non-empty strings")
            _require_json_value(item, f"{name}.{key}")
        return
    raise RecoveryExperimentError(f"{name} is not a canonical JSON value")


@dataclass(frozen=True)
class RecoveryEvent:
    """One ordered process-lifecycle observation."""

    sequence: int
    run_id: str
    attempt_id: str
    kind: str
    monotonic_ns: int
    completed_items: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("sequence", "monotonic_ns", "completed_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise RecoveryExperimentError(f"{name} must be non-negative")
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise RecoveryExperimentError("run_id is invalid")
        if (
            not isinstance(self.attempt_id, str)
            or _ATTEMPT_ID.fullmatch(self.attempt_id) is None
        ):
            raise RecoveryExperimentError("attempt_id is invalid")
        if self.kind not in _KINDS:
            raise RecoveryExperimentError("event kind is unsupported")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        _require_json_value(self.payload, "payload")
        json.dumps(self.payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "kind": self.kind,
            "monotonic_ns": self.monotonic_ns,
            "completed_items": self.completed_items,
            "payload": dict(self.payload),
        }


def chain_recovery_event(
    event: RecoveryEvent, previous_sha256: str | None = None
) -> dict[str, Any]:
    """Bind one event to the full accepted trace prefix."""

    if not isinstance(event, RecoveryEvent):
        raise TypeError("event must be RecoveryEvent")
    if previous_sha256 is not None and (
        not isinstance(previous_sha256, str)
        or _DIGEST.fullmatch(previous_sha256) is None
    ):
        raise RecoveryExperimentError("previous_sha256 must be a SHA-256 digest")
    payload = {**event.to_mapping(), "previous_sha256": previous_sha256}
    payload["event_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload
