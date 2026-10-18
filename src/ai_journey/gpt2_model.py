"""Independent GPT-2 decoder implementation for weight interoperability."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .gpt2_config import GPT2Config, GPT2ConfigError


def gelu_new(inputs: Tensor) -> Tensor:
    """Apply the tanh GELU approximation used by the published GPT-2 model."""

    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if not inputs.is_floating_point():
        raise TypeError("inputs must have a floating-point dtype")
    return (
        0.5
        * inputs
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * inputs.pow(3.0))
            )
        )
    )


class GPT2SelfAttention(nn.Module):
    """GPT-2 causal attention with fused query, key, and value projection."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        self.head_count = config.head_count
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("attention inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise GPT2ConfigError(
                "attention inputs must have non-empty (batch, time, channels) shape"
            )
        batch, time, channels = inputs.shape
        if channels != self.c_attn.in_features:
            raise GPT2ConfigError("attention channels must match embedding_dim")
        if time > self.context_length:
            raise GPT2ConfigError("attention sequence exceeds context_length")
        if inputs.dtype != self.c_attn.weight.dtype:
            raise GPT2ConfigError("attention inputs must match parameter dtype")
        if inputs.device != self.c_attn.weight.device:
            raise GPT2ConfigError("attention inputs must match parameter device")

        query, key, value = self.c_attn(inputs).split(channels, dim=2)
        head_shape = (batch, time, self.head_count, self.head_dim)
        query = query.view(head_shape).transpose(1, 2)
        key = key.view(head_shape).transpose(1, 2)
        value = value.view(head_shape).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(attended))
