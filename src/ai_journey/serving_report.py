"""Auditable reports for bounded serving evidence."""

from __future__ import annotations

from dataclasses import dataclass


class ServingReportError(ValueError):
    """Raised when a serving report is incomplete or unverifiable."""


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ServingReportError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class EvidenceReferences:
    """Content identities for every source behind one serving report."""

    contract_sha256: str
    trace_sha256: str
    observations_sha256: str
    recovery_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "contract_sha256",
            "trace_sha256",
            "observations_sha256",
            "recovery_sha256",
        ):
            _require_sha256(getattr(self, name), name)
