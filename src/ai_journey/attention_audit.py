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
