"""Replicated GPT-2 optimizer steps with explicit accumulation boundaries."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .ddp_metrics import token_nll
from .ddp_runtime import require_all_ranks


def accumulated_update(
    model: DistributedDataParallel, optimizer, batches, *, gradient_clip: float = 1.0
) -> dict:
    """Synchronize only the final micro-batch and perform one optimizer update."""
    if not isinstance(model, DistributedDataParallel):
        raise TypeError("model must be DistributedDataParallel")
    if (
        isinstance(gradient_clip, bool)
        or not isinstance(gradient_clip, (int, float))
        or not 0 < gradient_clip < float("inf")
    ):
        raise ValueError("gradient_clip must be positive and finite")
    batches = tuple(batches)
    valid = bool(batches)
    for inputs, targets in batches:
        valid = (
            valid
            and isinstance(inputs, torch.Tensor)
            and isinstance(targets, torch.Tensor)
        )
        if not valid:
            break
        valid = (
            inputs.dtype == targets.dtype == torch.long
            and inputs.ndim == 2
            and inputs.shape == targets.shape
            and 0 < inputs.shape[1] <= model.module.config.context_length
            and inputs.numel() > 0
        )
        if valid:
            valid = (
                min(inputs.min().item(), targets.min().item()) >= 0
                and max(inputs.max().item(), targets.max().item())
                < model.module.config.vocab_size
            )
    require_all_ranks(valid, phase="micro-batch validation")
    signature = (len(batches), tuple(batches[0][0].shape))
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    local_shapes_equal = len({tuple(inputs.shape) for inputs, _ in batches}) == 1
    require_all_ranks(
        local_shapes_equal and all(value == signature for value in signatures),
        phase="micro-batch schedule agreement",
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    tokens = 0
    model.train()
    for index, (inputs, targets) in enumerate(batches):
        context = model.no_sync() if index < len(batches) - 1 else nullcontext()
        with context:
            losses = token_nll(model(inputs), targets)
            (losses.mean() / len(batches)).backward()
            total_loss += losses.detach().double().sum().item()
            tokens += targets.numel()
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    require_all_ranks(finite, phase="finite gradient validation")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    ).item()
    optimizer.step()
    return {
        "loss_sum": total_loss,
        "tokens": tokens,
        "micro_batches": len(batches),
        "gradient_norm": norm,
        "clipped": norm > gradient_clip,
    }
