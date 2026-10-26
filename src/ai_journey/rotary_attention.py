"""Full-sequence causal self-attention with rotary query and key positions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .gpt2_config import GPT2Config, GPT2ConfigError
from .rope import apply_rotary_position_embeddings


class RotarySelfAttention(nn.Module):
    """GPT-2-shaped projections with rotary positions instead of learned ones."""

    def __init__(self, config: GPT2Config, *, base: float = 10_000.0) -> None:
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        if config.head_dim % 2:
            raise GPT2ConfigError("rotary head_dim must be even")
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            raise GPT2ConfigError("rotary base must be finite and greater than one")
        try:
            base_value = float(base)
        except OverflowError as exc:
            raise GPT2ConfigError(
                "rotary base must be finite and greater than one"
            ) from exc
        if not math.isfinite(base_value) or base_value <= 1:
            raise GPT2ConfigError("rotary base must be finite and greater than one")

        self.head_count = config.head_count
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.base = base_value
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Attend over a complete sequence without a cache or position table."""

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
        query, key = apply_rotary_position_embeddings(query, key, base=self.base)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(attended))

    def forward_with_cache(
        self,
        inputs: Tensor,
        *,
        past_key: Tensor | None = None,
        past_value: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Attend to new tokens and return extended rotated-key/value caches.

        Cache tensors are shaped ``(batch, heads, past_time, head_dim)`` and
        must be the unmodified outputs of an earlier call on this module.
        ``past_key`` already holds rotary positions; only new queries and keys
        are rotated at the absolute offset ``past_time``. Both cache tensors
        must be supplied together, or neither for prefill.
        """

        if not isinstance(inputs, Tensor):
            raise TypeError("attention inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise GPT2ConfigError(
                "attention inputs must have non-empty (batch, time, channels) shape"
            )
        batch, time, channels = inputs.shape
        if channels != self.c_attn.in_features:
            raise GPT2ConfigError("attention channels must match embedding_dim")
        if inputs.dtype != self.c_attn.weight.dtype:
            raise GPT2ConfigError("attention inputs must match parameter dtype")
        if inputs.device != self.c_attn.weight.device:
            raise GPT2ConfigError("attention inputs must match parameter device")
        if (past_key is None) != (past_value is None):
            raise GPT2ConfigError("past_key and past_value must be supplied together")

        past_time = 0
        if past_key is not None and past_value is not None:
            expected_prefix = (batch, self.head_count)
            for name, cache in (("past_key", past_key), ("past_value", past_value)):
                if not isinstance(cache, Tensor):
                    raise TypeError(f"{name} must be a torch.Tensor")
                if (
                    cache.ndim != 4
                    or cache.shape[:2] != expected_prefix
                    or cache.shape[-1] != self.head_dim
                ):
                    raise GPT2ConfigError(
                        f"{name} must have (batch, heads, past_time, head_dim) shape"
                    )
                if cache.dtype != inputs.dtype or cache.device != inputs.device:
                    raise GPT2ConfigError(f"{name} must match input dtype and device")
            if past_key.shape[2] != past_value.shape[2]:
                raise GPT2ConfigError("past_key and past_value must share past_time")
            past_time = past_key.shape[2]
        if past_time + time > self.context_length:
            raise GPT2ConfigError("attention sequence exceeds context_length")

        query, key, value = self.c_attn(inputs).split(channels, dim=2)
        head_shape = (batch, time, self.head_count, self.head_dim)
        query = query.view(head_shape).transpose(1, 2)
        key = key.view(head_shape).transpose(1, 2)
        value = value.view(head_shape).transpose(1, 2)
        query, key = apply_rotary_position_embeddings(
            query, key, position_offset=past_time, base=self.base
        )
        if past_key is not None and past_value is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[past_time : past_time + time, : past_time + time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(attended)), key, value
