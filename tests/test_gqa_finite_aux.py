"""Inactive dense objectives remain exact connected zeros at finite boundaries."""

import pytest
import torch
from torch.nn import functional as F

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gqa_config import GQAConfig
from ai_journey.training_models import build_training_model
from test_grouped_rotary_attention import config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_extreme_dense_logits_have_finite_zero_auxiliary_terms(
    monkeypatch, dtype
):
    model = build_training_model(
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 1)
    ).to(dtype=dtype)
    logits = torch.full(
        (1, 2, 256), torch.finfo(dtype).max / 2, dtype=dtype, requires_grad=True
    )
    monkeypatch.setattr(model.decoder, "forward", lambda _: logits)
    result = model.forward_training(torch.tensor([[1, 2]]))
    assert torch.isfinite(result.logits).all()
    ce = F.cross_entropy(result.logits.reshape(-1, 256), torch.tensor([2, 3]))
    assert torch.isfinite(ce)
    assert result.balance_loss.item() == result.router_z_loss.item() == 0.0
    assert result.balance_loss.requires_grad and result.router_z_loss.requires_grad
    (result.balance_loss + result.router_z_loss).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
