"""Bounded assistant-only supervised fine-tuning on the tiny CPU decoder."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from .sft_data import SFTDataError, SFTWindow


def _finite_cpu_model(model):
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if any(
        parameter.device.type != "cpu"
        or not parameter.is_floating_point()
        or not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise SFTDataError("SFT parameters must be finite floating-point CPU tensors")


def _validate_batches(batches):
    if (
        not isinstance(batches, tuple)
        or not batches
        or any(
            not isinstance(batch, tuple)
            or not batch
            or any(not isinstance(row, SFTWindow) for row in batch)
            for batch in batches
        )
    ):
        raise SFTDataError("SFT training requires nonempty immutable batches")
    batch_size = len(batches[0])
    width = len(batches[0][0].inputs)
    if any(
        len(batch) != batch_size or any(len(row.inputs) != width for row in batch)
        for batch in batches
    ):
        raise SFTDataError("SFT batches must have consistent rectangular shapes")


def train_sft_batches(model, batches, *, learning_rate):
    """Optimize exactly the supplied assistant-mask schedule."""
    _finite_cpu_model(model)
    _validate_batches(batches)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise SFTDataError("learning_rate must be finite and in (0, 0.1]")
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = (
        torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        if parameters
        else None
    )
    caller_modes = [(module, module.training) for module in model.modules()]
    losses = []
    input_tokens = supervised_tokens = 0
    try:
        model.train()
        for batch in batches:
            inputs = torch.tensor([row.inputs for row in batch], dtype=torch.long)
            targets = torch.tensor([row.targets for row in batch], dtype=torch.long)
            mask = torch.tensor([row.target_mask for row in batch], dtype=torch.bool)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with nullcontext() if optimizer is not None else torch.no_grad():
                _, loss = model(inputs, targets, loss_mask=mask)
            if loss is None or not bool(torch.isfinite(loss)):
                raise SFTDataError("SFT loss must remain finite")
            losses.append(float(loss.detach()))
            input_tokens += inputs.numel()
            supervised_tokens += int(mask.sum())
            if optimizer is not None:
                loss.backward()
                if any(
                    parameter.grad is None
                    or not bool(torch.isfinite(parameter.grad).all())
                    for parameter in parameters
                ):
                    raise SFTDataError(
                        "trainable SFT gradients must be present and finite"
                    )
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                _finite_cpu_model(model)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
    finally:
        for module, mode in caller_modes:
            module.training = mode
    return {
        "losses": losses,
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "optimizer_steps": len(losses) if optimizer is not None else 0,
    }
