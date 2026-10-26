"""Sparse execution must match the dense reference without invoking idle experts."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from ai_journey.moe_capacity import plan_capacity
from ai_journey.moe_dispatch import dispatch_experts
from ai_journey.moe_routing import top2_routes


def test_sparse_outputs_and_gradients_match_dense_expert_reference():
    torch.manual_seed(7)
    inputs = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    logits = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
    experts = nn.ModuleList([nn.Linear(3, 3).double() for _ in range(4)])
    dense_experts = deepcopy(experts)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    routes = top2_routes(logits)
    plan = plan_capacity(routes, capacity_factor=0.5)
    actual = dispatch_experts(inputs, experts, routes, plan)
    reference_routes = top2_routes(reference_logits)
    reference = torch.zeros_like(reference_inputs)
    for token in range(4):
        for choice in range(2):
            if plan.accepted[token, choice]:
                expert = int(reference_routes.expert_indices[token, choice])
                reference[0, token] = (
                    reference[0, token]
                    + dense_experts[expert](reference_inputs[0, token])
                    * reference_routes.weights[token, choice]
                )
    torch.testing.assert_close(actual, reference)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    for actual_param, expected in zip(experts.parameters(), dense_experts.parameters()):
        if expected.grad is None:
            assert actual_param.grad is None
        else:
            torch.testing.assert_close(actual_param.grad, expected.grad)


def test_idle_experts_are_not_executed_and_dropped_tokens_are_zero():
    class Unexpected(nn.Module):
        def forward(self, inputs):
            pytest.fail("idle expert must not execute")

    experts = [nn.Identity(), nn.Identity(), Unexpected(), Unexpected()]
    inputs = torch.ones(1, 3, 2)
    routes = top2_routes(torch.zeros(1, 3, 4))
    plan = plan_capacity(routes, capacity_factor=0.1)
    output = dispatch_experts(inputs, experts, routes, plan)
    assert output.tolist() == [[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]


def test_masked_inputs_are_not_dispatched():
    inputs = torch.tensor([[[2.0, 3.0], [10.0, 20.0]]])
    routes = top2_routes(torch.zeros(1, 2, 2), token_mask=torch.tensor([[True, False]]))
    output = dispatch_experts(
        inputs, [nn.Identity(), nn.Identity()], routes, plan_capacity(routes)
    )
    assert output.tolist() == [[[2.0, 3.0], [0.0, 0.0]]]


def test_expert_failure_does_not_silently_emit_nonfinite_values():
    class Broken(nn.Module):
        def forward(self, inputs):
            return inputs * float("nan")

    routes = top2_routes(torch.zeros(1, 1, 2))
    with pytest.raises(ValueError, match="expert outputs"):
        dispatch_experts(
            torch.ones(1, 1, 2),
            [Broken(), nn.Identity()],
            routes,
            plan_capacity(routes),
        )
