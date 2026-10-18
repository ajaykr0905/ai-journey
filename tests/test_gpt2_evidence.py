from __future__ import annotations

from copy import deepcopy

import pytest

from ai_journey.gpt2_audit import LogitParityResult
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_evidence import (
    GPT2EvidenceError,
    build_evidence,
    evidence_payload,
    validate_evidence_payload,
)
from ai_journey.gpt2_source import GPT2Source


def _config() -> GPT2Config:
    return GPT2Config(11, 4, 4, 2, 2, 1e-5, 0, 0, 0)


def _source() -> GPT2Source:
    return GPT2Source("openai/gpt2", "6" * 40, "c.json", "a" * 64, "w.bin", "b" * 64, 1)


def _parity() -> LogitParityResult:
    return LogitParityResult((1, 2), 22, 1e-5, 1e-6, 2, 2, "c" * 64, "d" * 64)


def test_builds_self_authenticating_complete_evidence() -> None:
    evidence = build_evidence(
        _source(),
        _config(),
        state_fingerprint="e" * 64,
        parameter_count=312,
        parity=_parity(),
        torch_version="2.14.1",
        python_version="3.13.9",
    )

    payload = evidence_payload(evidence)
    restored = validate_evidence_payload(payload)

    assert restored == evidence
    assert restored.compared_values == 22
    assert len(payload["evidence_sha256"]) == 64


def test_rejects_incomplete_parity_coverage() -> None:
    parity = LogitParityResult((1, 2), 21, 0.0, 0.0, 2, 2, "c" * 64, "d" * 64)

    with pytest.raises(GPT2EvidenceError, match="every probe vocabulary"):
        build_evidence(
            _source(),
            _config(),
            state_fingerprint="e" * 64,
            parameter_count=312,
            parity=parity,
            torch_version="2.14.1",
            python_version="3.13.9",
        )


def test_detects_changed_evidence_and_unknown_fields() -> None:
    evidence = build_evidence(
        _source(),
        _config(),
        state_fingerprint="e" * 64,
        parameter_count=312,
        parity=_parity(),
        torch_version="2.14.1",
        python_version="3.13.9",
    )
    changed = deepcopy(evidence_payload(evidence))
    changed["evidence"]["max_absolute_error"] = 0.5
    unknown = {**evidence_payload(evidence), "extra": True}

    with pytest.raises(GPT2EvidenceError, match="SHA-256 mismatch"):
        validate_evidence_payload(changed)
    with pytest.raises(GPT2EvidenceError, match="fields are not exact"):
        validate_evidence_payload(unknown)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("state_fingerprint", "not-a-digest", "SHA-256"),
        ("parameter_count", 0, "positive integer"),
        ("top1_matches", 3, "in \\[0"),
        ("mean_absolute_error", float("nan"), "finite"),
    ],
)
def test_rejects_invalid_evidence_values(
    field: str, value: object, message: str
) -> None:
    evidence = build_evidence(
        _source(),
        _config(),
        state_fingerprint="e" * 64,
        parameter_count=312,
        parity=_parity(),
        torch_version="2.14.1",
        python_version="3.13.9",
    )
    payload = evidence_payload(evidence)
    payload["evidence"][field] = value
    payload["evidence_sha256"] = evidence_payload(evidence)["evidence_sha256"]

    with pytest.raises(GPT2EvidenceError, match=message):
        validate_evidence_payload(payload)
