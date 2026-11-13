"""Strict controls for the Day 70 worker recovery experiment."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any


class RecoveryExperimentError(ValueError):
    """Raised when recovery evidence violates its declared contract."""


_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {
    "schema_version",
    "experiment_name",
    "work_items",
    "checkpoint_interval",
    "fault_after_items",
    "heartbeat_interval_items",
    "heartbeat_timeout_ms",
    "worker_timeout_ms",
}


@dataclass(frozen=True)
class RecoveryExperimentConfig:
    """Bounded, deterministic controls for one local process-recovery run."""

    experiment_name: str
    work_items: int
    checkpoint_interval: int
    fault_after_items: int
    heartbeat_interval_items: int
    heartbeat_timeout_ms: int
    worker_timeout_ms: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.experiment_name, str)
            or _NAME.fullmatch(self.experiment_name) is None
        ):
            raise RecoveryExperimentError("experiment_name must be a lowercase slug")
        for name in (
            "work_items",
            "checkpoint_interval",
            "fault_after_items",
            "heartbeat_interval_items",
            "heartbeat_timeout_ms",
            "worker_timeout_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise RecoveryExperimentError(f"{name} must be positive")
        if self.checkpoint_interval >= self.work_items:
            raise RecoveryExperimentError("checkpoint_interval must precede completion")
        if not self.checkpoint_interval < self.fault_after_items < self.work_items:
            raise RecoveryExperimentError(
                "fault_after_items must follow a checkpoint and precede completion"
            )
        if self.heartbeat_interval_items > self.checkpoint_interval:
            raise RecoveryExperimentError(
                "heartbeat_interval_items cannot exceed checkpoint_interval"
            )
        if self.heartbeat_timeout_ms >= self.worker_timeout_ms:
            raise RecoveryExperimentError(
                "heartbeat_timeout_ms must be lower than worker_timeout_ms"
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical versioned mapping."""

        return {"schema_version": 1, **asdict(self)}

    def fingerprint(self) -> str:
        """Bind every recovery control to a stable digest."""

        return sha256(
            json.dumps(
                self.to_mapping(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def config_from_mapping(payload: object) -> RecoveryExperimentConfig:
    """Parse a config without ignoring unknown controls."""

    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise RecoveryExperimentError("recovery config fields do not match schema")
    if payload["schema_version"] != 1:
        raise RecoveryExperimentError("recovery config schema version is unsupported")
    return RecoveryExperimentConfig(
        **{key: value for key, value in payload.items() if key != "schema_version"}
    )
