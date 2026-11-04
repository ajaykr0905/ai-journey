import pytest

from ai_journey.inference_seal import (
    inference_report_digest,
    seal_inference_report,
)


def test_report_seal_is_stable_across_mapping_order():
    first = seal_inference_report({"schema_version": 1, "nested": {"b": 2, "a": 1}})
    second = seal_inference_report({"nested": {"a": 1, "b": 2}, "schema_version": 1})

    assert first["report_sha256"] == second["report_sha256"]
    assert first["report_sha256"] == inference_report_digest(first)


def test_sealing_does_not_mutate_the_caller_report():
    report = {"schema_version": 1}

    sealed = seal_inference_report(report)

    assert "report_sha256" not in report
    assert len(sealed["report_sha256"]) == 64


def test_report_cannot_be_accidentally_resealed():
    with pytest.raises(ValueError, match="already sealed"):
        seal_inference_report(seal_inference_report({"schema_version": 1}))
