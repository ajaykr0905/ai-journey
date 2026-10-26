"""Sparse combination must not turn individually finite experts into infinity."""

import pytest
import torch
from torch import nn

from ai_journey.moe_capacity import plan_capacity
from ai_journey.moe_dispatch import dispatch_experts
from ai_journey.moe_routing import top2_routes


class ConstantExpert(nn.Module):
    def __init__(self, fraction=1.0):
        super().__init__()
        self.fraction = fraction

    def forward(self, inputs):
        return torch.full_like(inputs, torch.finfo(inputs.dtype).max * self.fraction)


def test_reject_finite_experts_whose_weighted_scatter_overflows():
    inputs = torch.ones(1, 1, 2)
    routes = top2_routes(torch.tensor([[[-9.949999809265137, 0.0]]]))
    experts = [ConstantExpert(), ConstantExpert()]
    assert all(bool(torch.isfinite(expert(inputs)).all()) for expert in experts)
    with pytest.raises(ValueError, match="combined expert outputs must be finite"):
        dispatch_experts(
            inputs, experts, routes, plan_capacity(routes, apply_capacity=False)
        )


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5])
def test_finite_combinations_preserve_existing_weighted_scatter(fraction):
    inputs = torch.ones(1, 1, 2)
    routes = top2_routes(torch.tensor([[[-9.949999809265137, 0.0]]]))
    result = dispatch_experts(
        inputs,
        [ConstantExpert(fraction), ConstantExpert(fraction)],
        routes,
        plan_capacity(routes, apply_capacity=False),
    )
    expected = torch.zeros_like(inputs)
    for weight in routes.weights[0]:
        expected += (
            torch.full_like(inputs, torch.finfo(inputs.dtype).max * fraction) * weight
        )
    assert bool(torch.isfinite(result).all())
    torch.testing.assert_close(result, expected)
