"""Expert firing evidence and auxiliary gradients describe the actual dispatch."""

import json

import pytest
import torch

from ai_journey.moe_layer import MoEFeedForward
from test_moe_config import small_config


@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_layer_losses_stats_and_actual_capacity_conserve_masked_routes(kind):
    layer = MoEFeedForward(small_config(mlp_kind=kind, capacity_factor=0.1))
    layer.router.projection.weight.data.zero_()
    output = layer.forward_training(
        torch.ones(1, 4, 8), token_mask=torch.tensor([[True, False, True, True]])
    )
    stats = output.stats
    assert stats["valid_tokens"] == 3 and stats["capacity"] == 1
    assert stats["attempted_counts"] == (3, 3, 0, 0)
    assert stats["accepted_counts"] == (1, 1, 0, 0)
    assert stats["dropped_counts"] == (2, 2, 0, 0)
    assert stats["accepted_per_token"] == ((2, 0, 0, 0),)
    assert stats["token_indices"] == (0, 2, 3)
    assert stats["selected_experts"] == ((0, 1), (0, 1), (0, 1))
    assert stats["accepted_routes"] == ((True, True), (False, False), (False, False))
    assert sum(stats["probability_sums"]) == 3
    assert output.idle_experts == (2, 3)
    assert json.loads(json.dumps(stats))["accepted_counts"] == [1, 1, 0, 0]
    (
        output.hidden.square().sum() + output.balance_loss + output.router_z_loss
    ).backward()
    assert torch.isfinite(layer.router.projection.weight.grad).all()
    assert layer.experts[2].parameters().__next__().grad is None


def test_auxiliary_objectives_reach_router_without_expert_backward():
    layer = MoEFeedForward(small_config()).double()
    with torch.no_grad():
        layer.router.projection.weight[0].fill_(1)
    output = layer.forward_training(torch.ones(1, 3, 8, dtype=torch.float64))
    (output.balance_loss + 0.01 * output.router_z_loss).backward()
    assert layer.router.projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in layer.experts.parameters())


def test_eval_drops_no_routes_and_does_not_cache_mutable_last_stats():
    layer = MoEFeedForward(small_config(capacity_factor=0.01)).eval()
    layer.router.projection.weight.data.zero_()
    output = layer.forward_training(torch.ones(1, 5, 8))
    assert output.stats["accepted_per_token"] == ((2, 2, 2, 2, 2),)
    assert output.stats["dropped_counts"] == (0, 0, 0, 0)
    assert not output.stats["capacity_applied"]
    assert not hasattr(layer, "last_stats") and not hasattr(layer, "last_aux_loss")


def test_all_masked_layer_emits_zero_auxiliary_loss_and_no_routes():
    layer = MoEFeedForward(small_config())
    output = layer.forward_training(
        torch.ones(1, 2, 8), token_mask=torch.zeros(1, 2, dtype=torch.bool)
    )
    assert output.hidden.count_nonzero() == 0
    assert output.balance_loss == output.router_z_loss == 0
    assert output.stats["accepted_per_token"] == ((0, 0),)
    assert output.idle_experts == (0, 1, 2, 3)
