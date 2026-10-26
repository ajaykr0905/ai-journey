"""Capacity plans conserve routes and never let padding consume admission."""

from dataclasses import replace

import pytest
import torch

from ai_journey.moe_capacity import plan_capacity
from ai_journey.moe_routing import top2_routes


def test_skewed_routes_have_deterministic_first_come_admission():
    routes = top2_routes(torch.zeros(1, 4, 4))
    plan = plan_capacity(routes, capacity_factor=0.5)
    assert plan.capacity == 1
    assert plan.accepted.tolist() == [
        [True, True],
        [False, False],
        [False, False],
        [False, False],
    ]
    assert plan.attempted_counts.tolist() == [4, 4, 0, 0]
    assert plan.accepted_counts.tolist() == [1, 1, 0, 0]
    assert plan.dropped_counts.tolist() == [3, 3, 0, 0]


def test_padding_does_not_change_valid_route_priority_or_capacity():
    logits = torch.zeros(1, 5, 4)
    routes = top2_routes(
        logits, token_mask=torch.tensor([[True, False, True, False, True]])
    )
    plan = plan_capacity(routes, capacity_factor=1)
    assert routes.token_indices.tolist() == [0, 2, 4]
    assert plan.capacity == 2
    assert plan.accepted.tolist() == [[True, True], [True, True], [False, False]]
    assert sum(plan.attempted_counts) == 6


def test_eval_is_dropless_even_when_all_tokens_choose_same_experts():
    plan = plan_capacity(
        top2_routes(torch.zeros(5, 4)), capacity_factor=0.01, apply_capacity=False
    )
    assert plan.accepted.all() and plan.capacity == 10
    assert plan.dropped_counts.tolist() == [0, 0, 0, 0]


def test_all_masked_plan_has_zero_counts_and_zero_capacity():
    routes = top2_routes(torch.zeros(2, 4), token_mask=torch.zeros(2, dtype=torch.bool))
    plan = plan_capacity(routes)
    assert plan.capacity == 0 and plan.accepted.shape == (0, 2)
    assert plan.attempted_counts.sum() == plan.accepted_counts.sum() == 0


@pytest.mark.parametrize(
    "tokens,experts,factor", [(1, 2, 1), (7, 4, 1.25), (9, 8, 0.5)]
)
def test_capacity_count_conservation(tokens, experts, factor):
    torch.manual_seed(5)
    plan = plan_capacity(
        top2_routes(torch.randn(tokens, experts)), capacity_factor=factor
    )
    assert plan.attempted_counts.sum() == tokens * 2
    assert torch.equal(
        plan.attempted_counts, plan.accepted_counts + plan.dropped_counts
    )
    assert (plan.accepted_counts <= plan.capacity).all()


@pytest.mark.parametrize(
    "corruption", ["duplicate_token", "duplicate_expert", "bad_weights"]
)
def test_forged_route_identity_is_rejected(corruption):
    routes = top2_routes(torch.zeros(2, 4))
    if corruption == "duplicate_token":
        routes = replace(routes, token_indices=torch.tensor([0, 0]))
    elif corruption == "duplicate_expert":
        routes = replace(routes, expert_indices=torch.zeros(2, 2, dtype=torch.long))
    else:
        routes = replace(routes, weights=torch.zeros(2, 2))
    with pytest.raises(ValueError):
        plan_capacity(routes)
