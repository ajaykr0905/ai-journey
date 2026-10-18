from __future__ import annotations

import pytest
import torch

from ai_journey.gpt2_audit import GPT2AuditError, audit_logit_parity
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def _model() -> GPT2LanguageModel:
    torch.manual_seed(45)
    return GPT2LanguageModel(
        GPT2Config(
            vocab_size=11,
            context_length=4,
            embedding_dim=4,
            head_count=2,
            layer_count=1,
            embedding_dropout=0,
            attention_dropout=0,
            residual_dropout=0,
        )
    )


def test_reports_exact_full_logit_parity_and_restores_model_mode() -> None:
    model = _model().train()
    token_ids = torch.tensor([[1, 2, 3]])
    model.eval()
    with torch.no_grad():
        reference = model(token_ids).clone()
    model.train()

    result = audit_logit_parity(model, token_ids, reference)

    assert result.compared_values == 33
    assert result.max_absolute_error == 0
    assert result.mean_absolute_error == 0
    assert result.top1_matches == result.top1_total == 3
    assert result.reference_digest == result.candidate_digest
    assert model.training


def test_detects_bounded_logit_drift_even_when_top1_is_unchanged() -> None:
    model = _model().eval()
    token_ids = torch.tensor([[1, 2]])
    with torch.no_grad():
        reference = model(token_ids).clone()
    reference[..., 0] += 1e-3

    result = audit_logit_parity(model, token_ids, reference)

    assert result.max_absolute_error == pytest.approx(1e-3, abs=1e-8)
    assert result.mean_absolute_error > 0
    assert result.reference_digest != result.candidate_digest


def test_rejects_nonfinite_or_wrong_shape_reference_logits() -> None:
    model = _model().eval()
    token_ids = torch.tensor([[1, 2]])
    nonfinite = torch.zeros(1, 2, 11)
    nonfinite[0, 0, 0] = float("nan")

    with pytest.raises(GPT2AuditError, match="finite"):
        audit_logit_parity(model, token_ids, nonfinite)
    with pytest.raises(GPT2AuditError, match="does not match"):
        audit_logit_parity(model, token_ids, torch.zeros(1, 2, 10))
