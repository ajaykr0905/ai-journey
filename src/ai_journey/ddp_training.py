"""Replicated GPT-2 optimizer steps with explicit accumulation boundaries."""

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from .ddp_metrics import token_nll
from .ddp_runtime import require_all_ranks


def accumulated_update(model: DistributedDataParallel, optimizer, batches) -> dict:
    """Synchronize only the final micro-batch and perform one optimizer update."""
    if not isinstance(model, DistributedDataParallel):
        raise TypeError("model must be DistributedDataParallel")
    batches = tuple(batches)
    if not batches:
        raise ValueError("at least one micro-batch is required")
    if len({tuple(inputs.shape) for inputs, _ in batches}) != 1:
        raise ValueError("accumulation requires equal micro-batch shapes")
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
    optimizer.step()
    return {"loss_sum": total_loss, "tokens": tokens, "micro_batches": len(batches)}
