"""Canonical progress checkpoints for worker recovery attempts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from .recovery_config import RecoveryExperimentError

_RUN_ID = re.compile(r"[a-z0-9-]+-[0-9a-f]{16}")
_ATTEMPT_ID = re.compile(r"attempt-[1-9][0-9]*-[0-9a-f]{16}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "completed_items",
    "workload_sha256",
    "accumulator_sha256",
    "checkpoint_sha256",
}


@dataclass(frozen=True)
class RecoveryCheckpoint:
    """Durable progress and output identity at an item boundary."""

    run_id: str
    attempt_id: str
    completed_items: int
    workload_sha256: str
    accumulator_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise RecoveryExperimentError("run_id is invalid")
        if (
            not isinstance(self.attempt_id, str)
            or _ATTEMPT_ID.fullmatch(self.attempt_id) is None
        ):
            raise RecoveryExperimentError("attempt_id is invalid")
        if isinstance(self.completed_items, bool) or not isinstance(
            self.completed_items, int
        ):
            raise TypeError("completed_items must be an integer")
        if self.completed_items < 0:
            raise RecoveryExperimentError("completed_items must be non-negative")
        for name in ("workload_sha256", "accumulator_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise RecoveryExperimentError(f"{name} must be a SHA-256 digest")

    def unsigned_mapping(self) -> dict[str, Any]:
        """Return the fields protected by the checkpoint seal."""

        return {"schema_version": 1, **asdict(self)}

    def to_mapping(self) -> dict[str, Any]:
        """Return a canonical, self-sealed checkpoint mapping."""

        payload = self.unsigned_mapping()
        payload["checkpoint_sha256"] = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload


def checkpoint_from_mapping(payload: object) -> RecoveryCheckpoint:
    """Validate exact fields and the checkpoint seal."""

    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise RecoveryExperimentError("checkpoint fields do not match schema")
    if payload["schema_version"] != 1:
        raise RecoveryExperimentError("checkpoint schema version is unsupported")
    unsigned = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    expected = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if payload["checkpoint_sha256"] != expected:
        raise RecoveryExperimentError("checkpoint digest does not match")
    return RecoveryCheckpoint(
        **{key: value for key, value in unsigned.items() if key != "schema_version"}
    )
