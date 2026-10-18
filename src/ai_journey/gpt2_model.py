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


class GPT2MLP(nn.Module):
    """GPT-2 position-wise feed-forward network."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        hidden_dim = 4 * config.embedding_dim
        self.c_fc = nn.Linear(config.embedding_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("MLP inputs must be a torch.Tensor")
        if inputs.ndim != 3 or inputs.shape[-1] != self.c_fc.in_features:
            raise GPT2ConfigError(
                "MLP inputs must have (batch, time, embedding_dim) shape"
            )
        if inputs.dtype != self.c_fc.weight.dtype:
            raise GPT2ConfigError("MLP inputs must match parameter dtype")
        if inputs.device != self.c_fc.weight.device:
            raise GPT2ConfigError("MLP inputs must match parameter device")
        return self.dropout(self.c_proj(gelu_new(self.c_fc(inputs))))


class GPT2Block(nn.Module):
    """One pre-normalized GPT-2 attention and MLP residual block."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        self.ln_1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attn(self.ln_1(inputs))
        return hidden + self.mlp(self.ln_2(hidden))


class GPT2LanguageModel(nn.Module):
    """Decoder-only GPT-2 with learned positions and tied token logits."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be GPT2Config")
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.wpe = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.embedding_dropout)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.layer_count)])
        self.ln_f = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.dtype != torch.long or token_ids.ndim != 2:
            raise TypeError("token_ids must be a rank-two torch.long tensor")
        batch, time = token_ids.shape
        if batch == 0 or time == 0:
            raise GPT2ConfigError("token batch and time dimensions must be non-empty")
        if time > self.config.context_length:
            raise GPT2ConfigError("token sequence exceeds context_length")
        if token_ids.device != self.wte.weight.device:
            raise GPT2ConfigError("token_ids must match the model device")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocab_size:
            raise GPT2ConfigError("token_ids contain an out-of-range id")
        positions = torch.arange(time, device=token_ids.device)
        hidden = self.drop(self.wte(token_ids) + self.wpe(positions))
        for block in self.h:
            hidden = block(hidden)
        return self.lm_head(self.ln_f(hidden))

    @property
    def parameter_count(self) -> int:
        """Count unique trainable values, excluding the tied head alias."""

        return sum(parameter.numel() for parameter in self.parameters())


def expected_gpt2_parameter_count(config: GPT2Config) -> int:
    """Calculate the exact unique parameter count without allocating a model."""

    if not isinstance(config, GPT2Config):
        raise TypeError("config must be GPT2Config")
    width = config.embedding_dim
    embeddings = width * (config.vocab_size + config.context_length)
    block = 12 * width**2 + 13 * width
    final_norm = 2 * width
    return embeddings + config.layer_count * block + final_norm
