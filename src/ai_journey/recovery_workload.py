"""Deterministic public-source work items for recovery experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from .recovery_config import RecoveryExperimentConfig, RecoveryExperimentError


@dataclass(frozen=True)
class RecoveryWorkItem:
    """One content-private unit of reproducible CPU work."""

    index: int
    payload_sha256: str


@dataclass(frozen=True)
class RecoveryWorkload:
    """Identity and ordered work items derived from bounded source bytes."""

    source_bytes: int
    source_sha256: str
    items: tuple[RecoveryWorkItem, ...]

    def fingerprint(self) -> str:
        """Bind source identity, source size, order, and every work item."""

        payload = {
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
            "items": [
                {"index": item.index, "payload_sha256": item.payload_sha256}
                for item in self.items
            ],
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def build_recovery_workload(
    source: bytes,
    config: RecoveryExperimentConfig,
    *,
    max_source_bytes: int = 1024 * 1024,
) -> RecoveryWorkload:
    """Create ordered item identities without exposing source content."""

    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if not isinstance(config, RecoveryExperimentConfig):
        raise TypeError("config must be RecoveryExperimentConfig")
    if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int):
        raise TypeError("max_source_bytes must be an integer")
    if max_source_bytes <= 0:
        raise RecoveryExperimentError("max_source_bytes must be positive")
    if not source or len(source) > max_source_bytes:
        raise RecoveryExperimentError("source must be non-empty and bounded")
    source_digest = sha256(source).hexdigest()
    items = tuple(
        RecoveryWorkItem(
            index=index,
            payload_sha256=sha256(
                source_digest.encode() + b":" + str(index).encode()
            ).hexdigest(),
        )
        for index in range(config.work_items)
    )
    return RecoveryWorkload(
        source_bytes=len(source), source_sha256=source_digest, items=items
    )
