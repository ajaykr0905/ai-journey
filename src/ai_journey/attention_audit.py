"""Independent, differentiable attention oracles for small CPU experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .transformer_lab import CausalSelfAttention, TransformerLabError


@dataclass(frozen=True)
class AttentionReference:
    output: Tensor
    weights: Tensor


@dataclass(frozen=True)
class AttentionGradientAudit:
    forward_error: float
    input_gradient_error: float
    qkv_weight_gradient_error: float
    qkv_bias_gradient_error: float
    projection_weight_gradient_error: float
    projection_bias_gradient_error: float

    @property
    def maximum_error(self) -> float:
        return max(vars(self).values())


@dataclass(frozen=True)
class AttentionCausalityAudit:
    prefix_error: float
    future_gradient: float
    boundaries_checked: int


def audit_attention_causality(
    attention: CausalSelfAttention, inputs: Tensor
) -> AttentionCausalityAudit:
    """Perturb each future suffix and differentiate every unchanged prefix."""
    per_head_reference(attention, inputs)
    original = attention(inputs).detach()
    prefix_error = 0.0
    future_gradient = 0.0
    for boundary in range(1, inputs.shape[1]):
        changed = inputs.detach().clone()
        suffix = changed[:, boundary:]
        # Channel-dependent perturbations avoid the constant-shift blind spot.
        suffix.add_(
            torch.arange(
                1, inputs.shape[-1] + 1, device=inputs.device, dtype=inputs.dtype
            )
            * 11
        )
        perturbed = attention(changed).detach()
        prefix_error = max(
            prefix_error,
            float((original[:, :boundary] - perturbed[:, :boundary]).abs().max()),
        )
        sample = inputs.detach().clone().requires_grad_(True)
        prefix = attention(sample)[:, :boundary]
        probe = torch.linspace(
            0.1, 1.0, prefix.numel(), device=inputs.device, dtype=inputs.dtype
        ).reshape_as(prefix)
        gradient = torch.autograd.grad((prefix * probe).sum(), sample)[0]
        future_gradient = max(
            future_gradient, float(gradient[:, boundary:].detach().abs().max())
        )
    return AttentionCausalityAudit(prefix_error, future_gradient, inputs.shape[1] - 1)


def audit_attention_gradients(
    attention: CausalSelfAttention, inputs: Tensor
) -> AttentionGradientAudit:
    """Compare all differentiable attention paths without changing caller grads."""
    sample = inputs.detach().clone().requires_grad_(True)
    reference = per_head_reference(attention, sample).output
    actual = attention(sample)
    probe = torch.linspace(
        0.1, 1.0, actual.numel(), dtype=actual.dtype, device=actual.device
    ).reshape_as(actual)
    parameters = (
        attention.query_key_value.weight,
        attention.query_key_value.bias,
        attention.projection.weight,
        attention.projection.bias,
    )
    if any(
        parameter is None or not parameter.requires_grad for parameter in parameters
    ):
        raise TransformerLabError(
            "gradient audit requires trainable projection weights and biases"
        )
    actual_gradients = torch.autograd.grad(
        (actual * probe).sum(), (sample, *parameters)
    )
    reference_gradients = torch.autograd.grad(
        (reference * probe).sum(), (sample, *parameters)
    )
    errors = [
        float((left - right).detach().abs().max())
        for left, right in zip(actual_gradients, reference_gradients, strict=True)
    ]
    return AttentionGradientAudit(
        float((actual - reference).detach().abs().max()), *errors
    )


def per_head_reference(
    attention: CausalSelfAttention, inputs: Tensor
) -> AttentionReference:
    """Compute each query/head independently, without packed batched attention.

    This slow oracle retains autograd and is restricted to evaluation mode.
    It uses the same learned parameters, including both projection biases.
    """
    if not isinstance(attention, CausalSelfAttention):
        raise TypeError("attention must be CausalSelfAttention")
    if any(module.training for module in attention.modules()):
        raise TransformerLabError("attention reference requires evaluation mode")
    if not isinstance(inputs, Tensor) or inputs.ndim != 3:
        raise TypeError("inputs must be a rank-three tensor")
    batch, time, width = inputs.shape
    if batch <= 0 or time <= 0 or width != attention.head_count * attention.head_dim:
        raise TransformerLabError("invalid attention input shape")
    if time > attention.causal_mask.shape[0]:
        raise TransformerLabError("sequence exceeds block_size")
    parameter = attention.query_key_value.weight
    if inputs.dtype != parameter.dtype or inputs.device != parameter.device:
        raise TransformerLabError("input dtype and device must match parameters")
    if not torch.isfinite(inputs).all():
        raise TransformerLabError("inputs must be finite")
    head_outputs = []
    batch_weights = []
    for sample in inputs:
        outputs = []
        weights = []
        for head in range(attention.head_count):
            start = head * attention.head_dim
            end = start + attention.head_dim
            projections = []
            for offset in (0, width, 2 * width):
                bias = attention.query_key_value.bias
                projections.append(
                    F.linear(
                        sample,
                        parameter[offset + start : offset + end],
                        None if bias is None else bias[offset + start : offset + end],
                    )
                )
            query, key, value = projections
            rows = []
            head_rows = []
            for position in range(time):
                scores = (
                    torch.stack(
                        [
                            (query[position] * key[past]).sum()
                            for past in range(position + 1)
                        ]
                    )
                    / attention.head_dim**0.5
                )
                probabilities = torch.softmax(scores, dim=0)
                rows.append(
                    torch.stack(
                        [
                            probabilities[past] * value[past]
                            for past in range(position + 1)
                        ]
                    ).sum(dim=0)
                )
                head_rows.append(F.pad(probabilities, (0, time - position - 1)))
            outputs.append(torch.stack(rows))
            weights.append(torch.stack(head_rows))
        head_outputs.append(torch.cat(outputs, dim=-1))
        batch_weights.append(torch.stack(weights))
    combined = torch.stack(head_outputs)
    return AttentionReference(
        F.linear(combined, attention.projection.weight, attention.projection.bias),
        torch.stack(batch_weights),
    )
