"""Opt-in decoder variants for controlled GPT-2 architecture comparisons."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .decoder_variant_config import DecoderVariantConfig
from .gpt2_config import GPT2ConfigError
from .gpt2_model import GPT2MLP, GPT2SelfAttention
from .modern_architecture import RMSNorm
from .rotary_attention import RotarySelfAttention
from .swiglu import SwiGLU


class DecoderVariantBlock(nn.Module):
    """A pre-normalized residual block selected by a validated variant contract."""

    def __init__(self, config: DecoderVariantConfig) -> None:
        super().__init__()
        if not isinstance(config, DecoderVariantConfig):
            raise TypeError("config must be DecoderVariantConfig")
        base = config.gpt2
        if config.norm_kind == "rmsnorm":
            self.ln_1 = RMSNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
            self.ln_2 = RMSNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
        else:
            self.ln_1 = nn.LayerNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
        self.attn = (
            RotarySelfAttention(base, base=config.rope_base)
            if config.position_kind == "rope"
            else GPT2SelfAttention(base)
        )
        self.mlp = (
            nn.Sequential(
                SwiGLU(base.embedding_dim, config.resolved_swiglu_hidden_dim),
                nn.Dropout(base.residual_dropout),
            )
            if config.mlp_kind == "swiglu"
            else GPT2MLP(base)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attn(self.ln_1(inputs))
        return hidden + self.mlp(self.ln_2(hidden))


class DecoderVariantModel(nn.Module):
    """A tied-embedding decoder with independently selectable modern components.

    This does not replace ``GPT2LanguageModel``: the existing model keeps its
    published weight-interoperability contract. The default variant has the
    same state-dict schema and exact forward behavior as that baseline.
    """

    def __init__(self, config: DecoderVariantConfig) -> None:
        super().__init__()
        if not isinstance(config, DecoderVariantConfig):
            raise TypeError("config must be DecoderVariantConfig")
        self.config = config
        base = config.gpt2
        self.wte = nn.Embedding(base.vocab_size, base.embedding_dim)
        self.wpe = (
            nn.Embedding(base.context_length, base.embedding_dim)
            if config.position_kind == "learned"
            else None
        )
        self.drop = nn.Dropout(base.embedding_dropout)
        self.h = nn.ModuleList(
            [DecoderVariantBlock(config) for _ in range(base.layer_count)]
        )
        self.ln_f = (
            RMSNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
            if config.norm_kind == "rmsnorm"
            else nn.LayerNorm(base.embedding_dim, eps=base.layer_norm_epsilon)
        )
        self.lm_head = nn.Linear(base.embedding_dim, base.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.dtype != torch.long or token_ids.ndim != 2:
            raise TypeError("token_ids must be a rank-two torch.long tensor")
        batch, time = token_ids.shape
        base = self.config.gpt2
        if batch == 0 or time == 0:
            raise GPT2ConfigError("token batch and time dimensions must be non-empty")
        if time > base.context_length:
            raise GPT2ConfigError("token sequence exceeds context_length")
        if token_ids.device != self.wte.weight.device:
            raise GPT2ConfigError("token_ids must match the model device")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= base.vocab_size:
            raise GPT2ConfigError("token_ids contain an out-of-range id")
        hidden = self.wte(token_ids)
        if self.wpe is not None:
            positions = torch.arange(time, device=token_ids.device)
            hidden = hidden + self.wpe(positions)
        hidden = self.drop(hidden)
        for block in self.h:
            hidden = block(hidden)
        return self.lm_head(self.ln_f(hidden))

    @property
    def parameter_count(self) -> int:
        """Count unique trainable values, excluding the tied output head."""

        return sum(parameter.numel() for parameter in self.parameters())
