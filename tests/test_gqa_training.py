"""Grouped training preserves the dense endpoint and expert transaction rules."""

from copy import deepcopy

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_model import DecoderVariantModel
from ai_journey.decoder_variant_optimizer import (
    DecoderOptimizerConfig,
    build_decoder_optimizer,
)
from ai_journey.decoder_variant_training import train_decoder_step
from ai_journey.gqa_config import GQAConfig
from ai_journey.gqa_training import build_gqa_optimizer, train_gqa_step
from ai_journey.moe_config import MoEConfig
from ai_journey.moe_training import validate_moe_optimizer
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_grouped_rotary_attention import config


def test_dense_mha_endpoint_update_matches_existing_trainer_tensor_for_tensor():
    base = DecoderVariantConfig(config(), position_kind="rope")
    grouped = build_training_model(GQAConfig(base, 4))
    reference = DecoderVariantModel(base)
    reference.load_state_dict(grouped.decoder.state_dict())
    controls = DecoderOptimizerConfig(0.02, 0.8, 0.95, 1e-7)
    optimizer = build_gqa_optimizer(grouped, learning_rate=0.002, config=controls)
    old_optimizer = build_decoder_optimizer(
        reference, learning_rate=0.002, config=controls
    )
    inputs, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    actual = train_gqa_step(grouped, optimizer, inputs, targets)
    expected = train_decoder_step(reference, old_optimizer, inputs, targets)
    assert actual["ce_loss"] == expected["loss"]
    assert actual["balance_loss"] == actual["router_z_loss"] == 0
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(
            grouped.decoder.state_dict()[name], tensor, rtol=0, atol=0
        )


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("sparse", [False, True])
def test_grouped_dense_and_expert_updates_have_valid_complete_optimizer_state(
    heads, sparse
):
    kind = MoEConfig if sparse else DecoderVariantConfig
    model = build_training_model(GQAConfig(kind(config(), position_kind="rope"), heads))
    optimizer = build_gqa_optimizer(model, learning_rate=0.001)
    before = training_model_fingerprint(model)
    result = train_gqa_step(
        model,
        optimizer,
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[2, 3, -100]]),
        token_mask=torch.tensor([[True, True, False]]),
        z_weight=0.001,
    )
    assert result["target_tokens"] == 2
    assert validate_moe_optimizer(model, optimizer) == 1
    assert training_model_fingerprint(model) != before
    assert len(optimizer.state) == len(tuple(model.parameters()))
    assert bool(result["routing"]) is sparse


def test_grouped_optimizer_failure_restores_weights_rng_gradients_and_modes(
    monkeypatch,
):
    model = build_training_model(
        GQAConfig(MoEConfig(config(), position_kind="rope"), 1)
    ).eval()
    model.h[0].ln_1.train()
    optimizer = build_gqa_optimizer(model, learning_rate=0.001)
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    before, rng = training_model_fingerprint(model), torch.get_rng_state().clone()
    before_optimizer = deepcopy(optimizer.state_dict())
    modes = [module.training for module in model.modules()]
    step = optimizer.step

    def fail():
        step()
        torch.rand(1)
        raise RuntimeError("injected grouped update failure")

    monkeypatch.setattr(optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="injected grouped"):
        train_gqa_step(model, optimizer, torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))
    assert training_model_fingerprint(model) == before
    assert optimizer.state_dict() == before_optimizer
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(first.grad, torch.ones_like(first))
    assert [module.training for module in model.modules()] == modes
