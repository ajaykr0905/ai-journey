"""Replicated GPT-2 optimizer steps with explicit accumulation boundaries."""

from contextlib import nullcontext
import copy
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .ddp_metrics import token_nll
from .ddp_runtime import require_all_ranks
from .ddp_runtime import (
    reduce_token_totals,
    require_model_consensus,
    require_optimizer_consensus,
)
from .ddp_data import byte_batches
from .ddp_contracts import integer
from .gpt2_training import learning_rate_at
from .training_models import base_model_config


def _accumulated_update(
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
    model_config = base_model_config(model.module)
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
            and 0 < inputs.shape[1] <= model_config.context_length
            and inputs.numel() > 0
        )
        if valid:
            valid = (
                min(inputs.min().item(), targets.min().item()) >= 0
                and max(inputs.max().item(), targets.max().item())
                < model_config.vocab_size
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
            loss = None
            batch_total = 0.0
            try:
                losses = token_nll(model(inputs), targets)
                loss = losses.mean() / len(batches)
                batch_total = losses.detach().double().sum().item()
                if not torch.isfinite(loss) or not math.isfinite(
                    total_loss + batch_total
                ):
                    loss = None
            except (ValueError, RuntimeError, TypeError):
                pass
            require_all_ranks(loss is not None, phase="forward loss validation")
            loss.backward()
            total_loss += batch_total
            tokens += targets.numel()
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    require_all_ranks(finite, phase="finite gradient validation")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip, error_if_nonfinite=True
    ).item()
    stepped = True
    try:
        optimizer.step()
        stepped = all(
            torch.isfinite(parameter).all().item() for parameter in model.parameters()
        )
    except (RuntimeError, ValueError):
        stepped = False
    require_all_ranks(stepped, phase="optimizer update")
    return {
        "loss_sum": total_loss,
        "tokens": tokens,
        "micro_batches": len(batches),
        "gradient_norm": norm,
        "clipped": norm > gradient_clip,
    }


def accumulated_update(
    model, optimizer, batches, *, gradient_clip=1.0, learning_rate=None
) -> dict:
    """Restore caller state if any rank rejects a synchronous optimizer update."""
    parameters = tuple(model.parameters())
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = torch.get_rng_state().clone()
    saved_gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    modes = [(module, module.training) for module in model.modules()]
    try:
        if learning_rate is not None:
            if (
                isinstance(learning_rate, bool)
                or not isinstance(learning_rate, (float, int))
                or not 0 <= learning_rate < float("inf")
            ):
                raise ValueError("learning_rate must be finite and nonnegative")
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        result = _accumulated_update(
            model, optimizer, batches, gradient_clip=gradient_clip
        )
        result["model_sha256"] = require_model_consensus(model.module)
        require_optimizer_consensus(optimizer)
        return result
    except Exception:
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        torch.set_rng_state(saved_rng)
        for parameter, gradient in zip(parameters, saved_gradients, strict=True):
            parameter.grad = gradient
        raise
    finally:
        for module, mode in modes:
            module.training = mode


def train_segment(
    model, optimizer, source: bytes, config, topology, *, start_step: int, steps: int
) -> list[dict]:
    """Reconstruct rank-local epoch cursors from the durable global update count."""
    integer(start_step, "start_step")
    integer(steps, "steps", 1)
    if start_step + steps > config.optimizer_steps:
        raise ValueError("segment exceeds the optimizer schedule")
    metrics = []
    for step in range(start_step, start_step + steps):
        batches, plan = byte_batches(
            source,
            sequence_length=config.sequence_length,
            topology=topology,
            micro_batch_size=config.micro_batch_size,
            accumulation_steps=config.accumulation_steps,
            seed=config.seed,
        )
        epoch, offset = divmod(step, plan.updates_per_epoch)
        if epoch:
            batches, _ = byte_batches(
                source,
                sequence_length=config.sequence_length,
                topology=topology,
                micro_batch_size=config.micro_batch_size,
                accumulation_steps=config.accumulation_steps,
                seed=config.seed,
                epoch=epoch,
            )
        learning_rate = learning_rate_at(step, config)
        begin = offset * config.accumulation_steps
        result = accumulated_update(
            model,
            optimizer,
            batches[begin : begin + config.accumulation_steps],
            gradient_clip=config.gradient_clip,
            learning_rate=learning_rate,
        )
        loss, tokens, mean = reduce_token_totals(result["loss_sum"], result["tokens"])
        metrics.append(
            {
                **result,
                "step": step + 1,
                "loss_sum": loss,
                "tokens": tokens,
                "mean_loss": mean,
                "learning_rate": learning_rate,
            }
        )
    return metrics
