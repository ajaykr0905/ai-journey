"""CPU grouped-query rotary attention with compact native key/value caches."""

import math

import torch
from torch import Tensor, nn

from .gpt2_config import GPT2Config
from .rope import apply_rotary_position_embeddings

MAX_GQA_SCORE_ELEMENTS = 8_000_000
MAX_GQA_NATIVE_CACHE_BYTES = 64 * 1024 * 1024


class GroupedRotarySelfAttention(nn.Module):
    """Query heads share groups of K/V heads; stored caches are never expanded."""

    def __init__(self, config: GPT2Config, *, kv_heads: int, base: float = 10000.0):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        if (
            type(kv_heads) is not int
            or not 1 <= kv_heads <= config.head_count
            or config.head_count % kv_heads
        ):
            raise ValueError("kv_heads must divide query head_count")
        if config.head_dim % 2:
            raise ValueError("grouped rotary attention requires an even head_dim")
        if type(base) not in (int, float):
            raise ValueError("rotary base must be finite and greater than one")
        try:
            base = float(base)
        except OverflowError as exc:
            raise ValueError("rotary base must be finite and greater than one") from exc
        if not math.isfinite(base) or base <= 1:
            raise ValueError("rotary base must be finite and greater than one")
        self.head_count, self.kv_heads, self.head_dim = (
            config.head_count,
            kv_heads,
            config.head_dim,
        )
        self.context_length, self.base = config.context_length, float(base)
        self.width, self.kv_width = config.embedding_dim, kv_heads * config.head_dim
        self.c_attn = nn.Linear(self.width, self.width + 2 * self.kv_width)
        self.c_proj = nn.Linear(self.width, self.width)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self._forward(inputs)[0]

    def forward_with_cache(
        self,
        inputs: Tensor,
        *,
        past_key: Tensor | None = None,
        past_value: Tensor | None = None,
    ):
        if any(module.training for module in self.modules()):
            raise ValueError("cached grouped attention requires evaluation mode")
        return self._forward(
            inputs, past_key=past_key, past_value=past_value, materialize_cache=True
        )

    def _forward(
        self, inputs, *, past_key=None, past_value=None, materialize_cache=False
    ):
        if (
            not isinstance(inputs, Tensor)
            or inputs.ndim != 3
            or any(size <= 0 for size in inputs.shape)
        ):
            raise ValueError("grouped attention requires nonempty batch/time/width")
        if (
            inputs.shape[-1] != self.width
            or inputs.dtype != self.c_attn.weight.dtype
            or inputs.device != self.c_attn.weight.device
            or inputs.device.type != "cpu"
            or inputs.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("grouped attention requires matching CPU floating inputs")
        if (past_key is None) != (past_value is None):
            raise ValueError("both compact key/value prefixes must be supplied")
        batch, time, _ = inputs.shape
        offset = 0
        if past_key is not None:
            for cache in (past_key, past_value):
                if (
                    not isinstance(cache, Tensor)
                    or cache.ndim != 4
                    or cache.shape[:2] != (batch, self.kv_heads)
                    or cache.shape[-1] != self.head_dim
                    or cache.dtype != inputs.dtype
                    or cache.device != inputs.device
                ):
                    raise ValueError(
                        "compact cache must match batch/KV-heads/dtype/device"
                    )
            if past_key.shape != past_value.shape:
                raise ValueError("compact key/value prefix lengths differ")
            offset = past_key.shape[2]
        total = offset + time
        if total > self.context_length:
            raise ValueError("grouped attention exceeds context_length")
        if batch * self.head_count * time * total > MAX_GQA_SCORE_ELEMENTS:
            raise ValueError("grouped attention exceeds the CPU score-work budget")
        if (
            2 * batch * self.kv_heads * total * self.head_dim * inputs.element_size()
            > MAX_GQA_NATIVE_CACHE_BYTES
        ):
            raise ValueError(
                "grouped attention exceeds the native KV-cache byte budget"
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("grouped attention inputs must be finite")
        if past_key is not None and any(
            not bool(torch.isfinite(cache).all()) for cache in (past_key, past_value)
        ):
            raise ValueError("compact cache values must be finite")
        query, key, value = self.c_attn(inputs).split(
            (self.width, self.kv_width, self.kv_width), dim=-1
        )
        query = query.view(batch, time, self.head_count, self.head_dim).transpose(1, 2)
        key = key.view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        if any(
            not bool(torch.isfinite(tensor).all()) for tensor in (query, key, value)
        ):
            raise ValueError("grouped attention projections must be finite")
        query, key = apply_rotary_position_embeddings(
            query, key, position_offset=offset, base=self.base
        )
        if past_key is not None:
            key, value = (
                torch.cat((past_key, key), 2),
                torch.cat((past_value, value), 2),
            )
        elif materialize_cache:
            # Even a singleton contiguous V view retains the packed QKV storage.
            # A clone gives the returned native cache its own compact backing.
            value = value.clone(memory_format=torch.contiguous_format)
        repeats = self.head_count // self.kv_heads
        expanded_key = key.repeat_interleave(repeats, dim=1)
        expanded_value = value.repeat_interleave(repeats, dim=1)
        scores = query @ expanded_key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("grouped attention scores must be finite")
        mask = torch.arange(total)[None, :] <= (offset + torch.arange(time))[:, None]
        weights = self.attn_dropout(
            scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)
        )
        attended = (
            (weights @ expanded_value)
            .transpose(1, 2)
            .contiguous()
            .view(batch, time, self.width)
        )
        output = self.resid_dropout(self.c_proj(attended))
        if not bool(torch.isfinite(output).all()):
            raise ValueError("grouped attention output must be finite")
        return output, key, value
