"""Auditable reports for bounded serving evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256

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


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def serving_report_payload(report: ServingEvidenceReport) -> dict[str, object]:
    """Return the JSON-compatible semantic payload for a serving report."""

    if not isinstance(report, ServingEvidenceReport):
        raise TypeError("report must be ServingEvidenceReport")
    snapshot = report.snapshot
    return {
        "boundary": asdict(report.boundary),
        "claims": list(report.claims),
        "comparison_sha256": report.comparison_sha256,
        "references": asdict(report.references),
        "report_version": report.report_version,
        "snapshot": {
            "engine_label": snapshot.engine_label,
            "failure_rate": _fraction_payload(snapshot.failure_rate),
            "goodput": _fraction_payload(snapshot.goodput),
            "key": asdict(snapshot.key),
            "output_tokens_per_second": _fraction_payload(
                snapshot.output_tokens_per_second
            ),
            "p95_ttft_ns": snapshot.p95_ttft_ns,
            "peak_memory_bytes": snapshot.peak_memory_bytes,
        },
    }


def canonical_serving_report_bytes(report: ServingEvidenceReport) -> bytes:
    """Serialize one report with stable keys, fractions, and whitespace."""

    return json.dumps(
        serving_report_payload(report),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SealedServingReport:
    """A serving report paired with its canonical content identity."""

    report: ServingEvidenceReport
    report_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.report, ServingEvidenceReport):
            raise TypeError("report must be ServingEvidenceReport")
        _require_sha256(self.report_sha256, "report_sha256")


def seal_serving_report(report: ServingEvidenceReport) -> SealedServingReport:
    """Seal one validated report with its canonical SHA-256 digest."""

    digest = sha256(canonical_serving_report_bytes(report)).hexdigest()
    return SealedServingReport(report=report, report_sha256=digest)
