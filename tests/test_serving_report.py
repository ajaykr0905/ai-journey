from fractions import Fraction

import pytest

from ai_journey.serving_compare import ComparisonKey, ServingSnapshot
from ai_journey.serving_contract import ClaimBoundary
from ai_journey.serving_report import (
    EvidenceReferences,
    ServingEvidenceReport,
    ServingReportError,
    validate_reported_claims,
)


def _digest(character: str) -> str:
    return character * 64


def _references() -> EvidenceReferences:
    return EvidenceReferences(_digest("a"), _digest("b"), _digest("c"), _digest("d"))


def test_evidence_references_bind_every_serving_source():
    references = _references()

    assert references.contract_sha256 == _digest("a")
    assert references.recovery_sha256 == _digest("d")


def test_evidence_references_reject_noncanonical_digest():
    with pytest.raises(ServingReportError, match="trace_sha256"):
        EvidenceReferences(_digest("a"), "B" * 64, _digest("c"), _digest("d"))


def test_reported_claims_must_stay_inside_declared_boundary():
    boundary = ClaimBoundary("cpu_fixture", ("determinism", "schema_validation"))

    assert validate_reported_claims(boundary, ("schema_validation",)) == (
        "schema_validation",
    )

    with pytest.raises(ServingReportError, match="outside"):
        validate_reported_claims(boundary, ("gpu_throughput",))


def _snapshot() -> ServingSnapshot:
    return ServingSnapshot(
        ComparisonKey(1, _digest("e"), _digest("f"), _digest("b")),
        "fixture-engine",
        100,
        Fraction(50),
        Fraction(9, 10),
        Fraction(1, 10),
        1024,
    )


def test_reported_claims_must_be_sorted_and_unique():
    boundary = ClaimBoundary("analytical", ("a", "b"))

    with pytest.raises(ServingReportError, match="sorted"):
        validate_reported_claims(boundary, ("b", "a"))


def test_serving_report_records_claim_bounded_snapshot():
    report = ServingEvidenceReport(
        1,
        _references(),
        ClaimBoundary("cpu_fixture", ("schema_validation",)),
        ("schema_validation",),
        _snapshot(),
        _digest("9"),
    )

    assert report.snapshot.engine_label == "fixture-engine"
    assert report.comparison_sha256 == _digest("9")


def test_serving_report_rejects_invalid_comparison_digest():
    with pytest.raises(ServingReportError, match="comparison_sha256"):
        ServingEvidenceReport(
            1,
            _references(),
            ClaimBoundary("analytical", ("schema_validation",)),
            ("schema_validation",),
            _snapshot(),
            "missing",
        )
