"""Auditable reports for bounded serving evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .serving_compare import ServingSnapshot
from .serving_contract import ClaimBoundary


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


def validate_reported_claims(
    boundary: ClaimBoundary, claims: tuple[str, ...]
) -> tuple[str, ...]:
    """Require report claims to be canonical and explicitly authorized."""

    if not isinstance(boundary, ClaimBoundary):
        raise TypeError("boundary must be ClaimBoundary")
    if (
        not isinstance(claims, tuple)
        or not claims
        or any(not isinstance(claim, str) or not claim for claim in claims)
    ):
        raise TypeError("claims must be a nonempty text tuple")
    if claims != tuple(sorted(set(claims))):
        raise ServingReportError("claims must be sorted and unique")
    unsupported = set(claims) - set(boundary.allowed_claims)
    if unsupported:
        raise ServingReportError("report contains claims outside its boundary")
    return claims


@dataclass(frozen=True)
class ServingEvidenceReport:
    """A claim-bounded serving snapshot linked to immutable raw evidence."""

    report_version: int
    references: EvidenceReferences
    boundary: ClaimBoundary
    claims: tuple[str, ...]
    snapshot: ServingSnapshot
    comparison_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.report_version, bool)
            or not isinstance(self.report_version, int)
            or self.report_version <= 0
        ):
            raise ServingReportError("report_version must be positive")
        if not isinstance(self.references, EvidenceReferences):
            raise TypeError("references must be EvidenceReferences")
        if not isinstance(self.snapshot, ServingSnapshot):
            raise TypeError("snapshot must be ServingSnapshot")
        validate_reported_claims(self.boundary, self.claims)
        if self.comparison_sha256 is not None:
            _require_sha256(self.comparison_sha256, "comparison_sha256")
