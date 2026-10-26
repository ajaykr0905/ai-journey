"""Deterministic first-come expert admission, separate from gate gradients."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .moe_routing import MAX_ROUTER_ENTRIES, RoutingOutput


@dataclass(frozen=True)
class CapacityPlan:
    accepted: Tensor
    capacity: int
    attempted_counts: Tensor
    accepted_counts: Tensor
    dropped_counts: Tensor
    capacity_applied: bool


def validate_routes(routes: RoutingOutput) -> None:
    if not isinstance(routes, RoutingOutput):
        raise TypeError("routes must be RoutingOutput")
    ids, indices = routes.expert_indices, routes.token_indices
    if (
        not isinstance(ids, Tensor)
        or ids.dtype != torch.long
        or ids.device.type != "cpu"
        or ids.ndim != 2
        or ids.shape[1] != 2
        or indices.dtype != torch.long
        or indices.device.type != "cpu"
        or indices.ndim != 1
        or len(indices) != len(ids)
        or not routes.token_shape
        or any(type(v) is not int or v <= 0 for v in routes.token_shape)
    ):
        raise ValueError("route token and expert dimensions are invalid")
    size = math.prod(routes.token_shape)
    if len(indices) and (
        indices.min() < 0
        or indices.max() >= size
        or not bool((indices[1:] > indices[:-1]).all())
    ):
        raise ValueError("route token indices must be unique sorted in-range positions")
    probabilities = routes.probabilities
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(ids)
        or not 2 <= probabilities.shape[1] <= 64
    ):
        raise ValueError("route probability dimensions are invalid")
    experts = probabilities.shape[1]
    if probabilities.numel() > MAX_ROUTER_ENTRIES:
        raise ValueError("route probability work exceeds its budget")
    if len(ids) and (
        ids.min() < 0 or ids.max() >= experts or bool((ids[:, 0] == ids[:, 1]).any())
    ):
        raise ValueError("routes must name two distinct in-range experts")
    for values, expected in (
        (routes.weights, ids.shape),
        (probabilities, probabilities.shape),
        (routes.logits, probabilities.shape),
    ):
        if (
            values.shape != expected
            or values.device.type != "cpu"
            or values.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError(
                "route floating tensors must be finite matching CPU matrices"
            )
    for values in (routes.weights, probabilities):
        if bool((values < 0).any()) or not torch.allclose(
            values.sum(-1),
            torch.ones(len(ids), dtype=values.dtype),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("route weights and probabilities must be normalized")


def plan_capacity(
    routes: RoutingOutput,
    *,
    capacity_factor: float = 1.25,
    min_capacity: int = 1,
    apply_capacity: bool = True,
) -> CapacityPlan:
    """Admit token-major routes, top-one before top-two within a token.

    Capacity is max(min_capacity, ceil(factor * valid_tokens * 2 / experts)).
    An empty valid set has zero capacity. Dropless evaluation records capacity
    2*valid_tokens, sufficient for every route even under maximally skewed gates.
    """
    validate_routes(routes)
    if type(capacity_factor) not in (int, float) or not 0 < capacity_factor <= 16:
        raise ValueError("capacity_factor must be in (0, 16]")
    if type(min_capacity) is not int or not 1 <= min_capacity <= 1024:
        raise ValueError("min_capacity must be in [1, 1024]")
    if type(apply_capacity) is not bool:
        raise ValueError("apply_capacity must be boolean")
    tokens, experts = len(routes.token_indices), routes.probabilities.shape[1]
    capacity = (
        (
            max(min_capacity, math.ceil(capacity_factor * tokens * 2 / experts))
            if apply_capacity
            else tokens * 2
        )
        if tokens
        else 0
    )
    flat = routes.expert_indices.reshape(-1)
    accepted = torch.zeros_like(flat, dtype=torch.bool)
    for expert in range(experts):
        positions = (flat == expert).nonzero().flatten()
        accepted[positions[:capacity]] = True
    attempted = torch.bincount(flat, minlength=experts)
    counts = torch.bincount(flat[accepted], minlength=experts)
    return CapacityPlan(
        accepted.reshape(-1, 2),
        capacity,
        attempted,
        counts,
        attempted - counts,
        apply_capacity,
    )
