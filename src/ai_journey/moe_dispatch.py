"""Sparse CPU expert execution with differentiable weighted scatter-add."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .moe_capacity import CapacityPlan, validate_routes
from .moe_routing import RoutingOutput


def dispatch_experts(
    inputs: Tensor,
    experts: Sequence[nn.Module],
    routes: RoutingOutput,
    plan: CapacityPlan,
) -> Tensor:
    """Run accepted rows only; no dense all-expert forward is performed.

    Experts receive (1, selected_rows, width), preserving the existing GELU and
    SwiGLU module contracts. Dropped routes contribute zero; retained weights
    keep their pre-capacity values, including when only one route survives.
    """
    validate_routes(routes)
    if (
        not isinstance(inputs, Tensor)
        or inputs.ndim not in (2, 3)
        or tuple(inputs.shape[:-1]) != routes.token_shape
        or inputs.dtype != routes.weights.dtype
        or inputs.device.type != "cpu"
    ):
        raise ValueError("dispatch inputs must match route token shape and CPU dtype")
    if len(experts) != routes.probabilities.shape[1] or not all(
        isinstance(expert, nn.Module) for expert in experts
    ):
        raise ValueError("one expert module is required for each router expert")
    if (
        not isinstance(plan, CapacityPlan)
        or plan.accepted.shape != routes.expert_indices.shape
        or plan.accepted.dtype != torch.bool
        or plan.accepted.device.type != "cpu"
    ):
        raise ValueError("dispatch capacity mask does not match routes")
    actual = torch.bincount(
        routes.expert_indices[plan.accepted], minlength=len(experts)
    )
    if not torch.equal(actual, plan.accepted_counts) or bool(
        (actual > plan.capacity).any()
    ):
        raise ValueError("capacity mask and expert counts disagree")
    flat = inputs.reshape(-1, inputs.shape[-1])
    result = torch.zeros_like(flat)
    for expert_index, expert in enumerate(experts):
        pairs = ((routes.expert_indices == expert_index) & plan.accepted).nonzero()
        if not len(pairs):
            continue
        rows, choices = pairs.unbind(1)
        token_indices = routes.token_indices.index_select(0, rows)
        selected = flat.index_select(0, token_indices)
        outputs = expert(selected.unsqueeze(0)).squeeze(0)
        if (
            outputs.shape != selected.shape
            or outputs.dtype != selected.dtype
            or outputs.device != selected.device
            or not bool(torch.isfinite(outputs).all())
        ):
            raise ValueError(
                "expert outputs must be finite and match input shape/dtype/device"
            )
        weighted = outputs * routes.weights[rows, choices].unsqueeze(-1)
        result.index_add_(0, token_indices, weighted)
    return result.reshape_as(inputs)
