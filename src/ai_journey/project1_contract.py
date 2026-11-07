"""Immutable contracts for bounded Project 1 training runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256


class Project1RunError(ValueError):
    """Raised when Project 1 run evidence violates its contract."""


_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")


@dataclass(frozen=True)
class Project1RunContract:
    """Controls that identify and bound one public-data training run."""

    run_name: str
    spec_sha256: str
    source_sha256: str
    max_source_bytes: int
    max_parameters: int
    telemetry_interval_steps: int
    checkpoint_interval_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_name, str)
            or _RUN_NAME.fullmatch(self.run_name) is None
        ):
            raise Project1RunError("run_name must be a lowercase slug")
        for name in ("spec_sha256", "source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise Project1RunError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "max_source_bytes",
            "max_parameters",
            "telemetry_interval_steps",
            "checkpoint_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise Project1RunError(f"{name} must be positive")

    def to_mapping(self) -> dict[str, object]:
        """Return the versioned canonical contract mapping."""

        return {"schema_version": 1, **asdict(self)}

    def fingerprint(self) -> str:
        """Bind every launch control to one stable identity."""

        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()
