"""Finite, inspectable optimizer transitions for controlled decoder trials."""

import copy
import math

import torch
from torch.nn import functional as F

from .decoder_variant_runtime import isolated_decoder_runtime


def require_finite_decoder_state(value) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("decoder state contains non-finite tensors")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite_decoder_state(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_finite_decoder_state(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("decoder state contains a non-finite number")


@isolated_decoder_runtime()
def train_decoder_step(model, optimizer, inputs, targets, *, max_gradient_norm=1.0):
    """Validate gradients before update and roll back a non-finite optimizer step."""

    if (
        isinstance(max_gradient_norm, bool)
        or not isinstance(max_gradient_norm, (int, float))
        or not math.isfinite(max_gradient_norm)
        or max_gradient_norm <= 0
    ):
        raise ValueError("max_gradient_norm must be finite and positive")
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    bound = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if len(bound) != len(parameters) or {id(parameter) for parameter in bound} != {
        id(parameter) for parameter in parameters
    }:
        raise ValueError(
            "optimizer must bind each trainable decoder parameter exactly once"
        )
    if (
        inputs.dtype != torch.long
        or inputs.ndim != 2
        or targets.dtype != torch.long
        or targets.shape != inputs.shape
        or inputs.numel() == 0
    ):
        raise ValueError(
            "training requires matching nonempty rank-two long inputs and targets"
        )
    require_finite_decoder_state(model.state_dict())
    require_finite_decoder_state(optimizer.state_dict())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("decoder training loss is non-finite")
        loss.backward()
        if any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise FloatingPointError("decoder gradients must be present and finite")
        before = math.sqrt(
            math.fsum(
                float(parameter.grad.double().square().sum())
                for parameter in parameters
            )
        )
        torch.nn.utils.clip_grad_norm_(
            parameters, max_gradient_norm, error_if_nonfinite=True
        )
        after = math.sqrt(
            math.fsum(
                float(parameter.grad.double().square().sum())
                for parameter in parameters
            )
        )
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        try:
            optimizer.step()
            require_finite_decoder_state(model.state_dict())
            require_finite_decoder_state(optimizer.state_dict())
        except Exception:
            model.load_state_dict(saved_model, strict=True)
            optimizer.load_state_dict(saved_optimizer)
            raise
        return {
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": before,
            "gradient_norm_after_clip": after,
            "clipped": before > max_gradient_norm,
            "target_tokens": targets.numel(),
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
