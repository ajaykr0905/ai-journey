"""Exact regression decisions for comparable serving evidence."""

from __future__ import annotations

from dataclasses import dataclass


class ServingComparisonError(ValueError):
    """Raised when serving evidence cannot be compared safely."""


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ServingComparisonError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ComparisonKey:
    """Identity fields that must match across baseline and candidate runs."""

    protocol_version: int
    workload_sha256: str
    model_sha256: str
    trace_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ServingComparisonError("protocol_version must be positive")
        for name in ("workload_sha256", "model_sha256", "trace_sha256"):
            _require_sha256(getattr(self, name), name)
