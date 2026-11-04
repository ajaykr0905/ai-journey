import pytest

from ai_journey.serving_contract import ClaimBoundary
from ai_journey.serving_report import (
    EvidenceReferences,
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


def test_reported_claims_must_be_sorted_and_unique():
    boundary = ClaimBoundary("analytical", ("a", "b"))

    with pytest.raises(ServingReportError, match="sorted"):
        validate_reported_claims(boundary, ("b", "a"))
