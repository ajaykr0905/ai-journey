import pytest

from ai_journey.serving_report import (
    EvidenceReferences,
    ServingReportError,
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
