"""Token-weighted reductions used by replicated training and evaluation."""

import math

import torch

from .ddp_contracts import integer


def ddp_loss_scale(local_tokens: int, global_tokens: int, world_size: int) -> float:
    """Compensate for DDP's rank averaging when local target counts differ."""
    integer(local_tokens, "local_tokens")
    integer(global_tokens, "global_tokens", 1)
    integer(world_size, "world_size", 1)
    if local_tokens > global_tokens:
        raise ValueError("local_tokens exceeds global_tokens")
    return world_size * local_tokens / global_tokens


def weighted_mean(totals: list[tuple[float, int]]) -> float:
    if not totals:
        raise ValueError("at least one total is required")
    for loss_sum, tokens in totals:
        integer(tokens, "tokens")
        if isinstance(loss_sum, bool) or not math.isfinite(loss_sum) or loss_sum < 0:
            raise ValueError("loss sums must be finite and nonnegative")
    count = sum(tokens for _, tokens in totals)
    if count == 0:
        raise ValueError("at least one target token is required")
    return math.fsum(loss for loss, _ in totals) / count


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute per-target stable cross entropy without reducing padding weights."""
    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or targets.dtype != torch.long
    ):
        raise ValueError("logits and targets must have matching batch/time dimensions")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point values")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[-1]:
        raise ValueError("targets must be nonempty valid vocabulary ids")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
