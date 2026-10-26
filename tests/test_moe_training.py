"""MoE updates must preserve strong optimizer and failure-transaction contracts."""

from copy import deepcopy

import pytest
import torch

from ai_journey.decoder_variant_optimizer import (
    DecoderOptimizerConfig,
    build_decoder_optimizer,
)
from ai_journey.moe_training import (
    build_moe_optimizer,
    train_moe_step,
    validate_moe_optimizer,
)
from ai_journey.moe_route_audit import validate_moe_routing
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_moe_config import small_config


def state():
    model = build_training_model(small_config(capacity_factor=0.1))
    model.h[0].mlp.router.projection.weight.data.zero_()
    return model, build_moe_optimizer(model, learning_rate=0.001)


def test_idle_experts_receive_dense_adamw_counters_without_expert_execution():
    model, optimizer = state()
    result = train_moe_step(
        model, optimizer, torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    )
    assert validate_moe_optimizer(model, optimizer) == 1
    assert len(optimizer.state) == len(tuple(model.parameters()))
    assert result["target_tokens"] == 3
    assert result["objective"] == pytest.approx(
        result["ce_loss"] + 0.01 * result["balance_loss"]
    )
    for stats in result["routing"]:
        validate_moe_routing(stats, expert_count=4)
    assert result["routing"][0]["accepted_counts"] == (1, 1, 0, 0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_builder_matches_explicit_dense_comparison_policy():
    model, _ = state()
    config = DecoderOptimizerConfig(0.02, 0.8, 0.95, 1e-7)
    actual = build_moe_optimizer(model, learning_rate=0.003, config=config).state_dict()
    expected = build_decoder_optimizer(
        model, learning_rate=0.003, config=config
    ).state_dict()
    assert actual == expected


def test_failure_restores_weights_optimizer_rng_gradients_and_mixed_modes(monkeypatch):
    model, optimizer = state()
    model.eval()
    model.h[0].train()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    before = training_model_fingerprint(model)
    original_optimizer = deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    modes = [module.training for module in model.modules()]
    original = optimizer.step

    def fail():
        original()
        torch.rand(1)
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        train_moe_step(model, optimizer, torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))
    assert training_model_fingerprint(model) == before
    assert optimizer.state_dict() == original_optimizer
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert [module.training for module in model.modules()] == modes


def test_masked_targets_do_not_enter_loss_or_capacity():
    model, optimizer = state()
    result = train_moe_step(
        model,
        optimizer,
        torch.tensor([[1, 0, 2]]),
        torch.tensor([[2, -100, 3]]),
        token_mask=torch.tensor([[True, False, True]]),
    )
    assert result["target_tokens"] == result["routing"][0]["valid_tokens"] == 2
    assert result["routing"][0]["accepted_per_token"][0][1] == 0


def test_missing_active_gradient_is_not_excused_as_idle(monkeypatch):
    model, optimizer = state()
    model.ln_f.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    before = training_model_fingerprint(model)
    with pytest.raises(ValueError, match="active model gradients"):
        train_moe_step(model, optimizer, torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))
    assert training_model_fingerprint(model) == before


@pytest.mark.parametrize(
    "name,value",
    [("balance_weight", True), ("z_weight", 10**1000), ("max_gradient_norm", 0)],
)
def test_invalid_objective_controls_fail_before_update(name, value):
    model, optimizer = state()
    with pytest.raises(ValueError):
        train_moe_step(
            model, optimizer, torch.tensor([[1]]), torch.tensor([[2]]), **{name: value}
        )
