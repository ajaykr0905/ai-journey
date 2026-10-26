"""Deterministic CPU top-two routing with padding excluded before projection."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

MAX_ROUTER_ENTRIES = 2_000_000


@dataclass(frozen=True)
class RoutingOutput:
    token_shape: tuple[int, ...]
    token_indices: Tensor
    expert_indices: Tensor
    weights: Tensor
    probabilities: Tensor
    logits: Tensor


def eligible_tokens(values: Tensor, token_mask: Tensor | None):
    """Validate CPU floating inputs and return only eligible flattened rows."""
    if not isinstance(values, Tensor) or values.ndim not in (2, 3):
        raise ValueError("routing values must be rank-two or rank-three tensors")
    if values.device.type != "cpu" or values.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("reference routing requires CPU float32 or float64 values")
    if any(size <= 0 for size in values.shape):
        raise ValueError("routing dimensions must be nonempty")
    shape = tuple(values.shape[:-1])
    if token_mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=values.device)
    else:
        if (
            not isinstance(token_mask, Tensor)
            or token_mask.dtype != torch.bool
            or tuple(token_mask.shape) != shape
            or token_mask.device != values.device
        ):
            raise ValueError("token_mask must be matching boolean token dimensions")
        mask = token_mask
    indices = mask.reshape(-1).nonzero().flatten()
    selected = values.reshape(-1, values.shape[-1]).index_select(0, indices)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("eligible routing values must be finite")
    return shape, indices, selected


def top2_routes(logits: Tensor, *, token_mask: Tensor | None = None) -> RoutingOutput:
    """Select distinct experts; exact ties prefer lower expert IDs.

    Softmax probabilities over all experts remain available for balancing.
    Selected weights use a second stable softmax, avoiding division by a
    potentially underflowed selected-probability sum.
    """
    shape, indices, selected = eligible_tokens(logits, token_mask)
    experts = logits.shape[-1]
    if not 2 <= experts <= 64:
        raise ValueError("routing requires between two and 64 experts")
    if selected.numel() > MAX_ROUTER_ENTRIES:
        raise ValueError("routing exceeds the CPU router-entry budget")
    chosen = torch.argsort(selected, dim=-1, descending=True, stable=True)[:, :2]
    probabilities = torch.softmax(selected, dim=-1)
    weights = torch.softmax(selected.gather(1, chosen), dim=-1)
    return RoutingOutput(shape, indices, chosen, weights, probabilities, selected)


class Top2Router(nn.Module):
    """A bias-free projection whose masked rows never consume routing work."""

    def __init__(self, width: int, expert_count: int):
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("router width must be positive")
        if type(expert_count) is not int or not 2 <= expert_count <= 64:
            raise ValueError("expert_count must be in [2, 64]")
        self.projection = nn.Linear(width, expert_count, bias=False)

    def forward(self, inputs: Tensor, *, token_mask: Tensor | None = None):
        shape, indices, selected = eligible_tokens(inputs, token_mask)
        if selected.shape[-1] != self.projection.in_features:
            raise ValueError("router input width does not match")
        if inputs.dtype != self.projection.weight.dtype:
            raise ValueError("router input dtype does not match its weights")
        if (
            math.prod(shape) > MAX_ROUTER_ENTRIES
            or selected.shape[0] * self.projection.out_features > MAX_ROUTER_ENTRIES
        ):
            raise ValueError("routing exceeds the CPU router-entry budget")
        logits = self.projection(selected)
        # The empty eligible set is valid for padded evaluation chunks.
        if not len(indices):
            empty_ids = torch.empty((0, 2), dtype=torch.long)
            return RoutingOutput(
                shape, indices, empty_ids, logits[:, :2], logits, logits
            )
        routes = top2_routes(logits)
        return RoutingOutput(
            shape,
            indices,
            routes.expert_indices,
            routes.weights,
            routes.probabilities,
            routes.logits,
        )
