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

LayerKeyValue = tuple[Tensor, Tensor]
PastKeyValues = tuple[LayerKeyValue, ...]


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

    def forward_with_cache(
        self,
        token_ids: Tensor,
        *,
        past_key_values: PastKeyValues | None = None,
    ) -> tuple[Tensor, PastKeyValues]:
        """Return new-token logits and per-layer key/value caches for RoPE.

        Each cache pair contains already-rotated keys and unrotated values in
        ``(batch, heads, past_time, head_dim)`` order. Pass the returned tuple
        unchanged on the next call; no past cache tensor is modified in place.
        Learned-position variants retain their ordinary full-sequence forward
        path and do not expose cache semantics here.
        """

        if self.config.position_kind != "rope":
            raise GPT2ConfigError("cached forward requires a RoPE position variant")
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.dtype != torch.long or token_ids.ndim != 2:
            raise TypeError("token_ids must be a rank-two torch.long tensor")
        batch, time = token_ids.shape
        base = self.config.gpt2
        if batch == 0 or time == 0:
            raise GPT2ConfigError("token batch and time dimensions must be non-empty")
        if token_ids.device != self.wte.weight.device:
            raise GPT2ConfigError("token_ids must match the model device")

        past_time = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple) or len(past_key_values) != len(
                self.h
            ):
                raise GPT2ConfigError("past_key_values must have one pair per layer")
            expected_shape = (batch, base.head_count, base.head_dim)
            for layer_index, pair in enumerate(past_key_values):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise GPT2ConfigError(
                        f"past_key_values[{layer_index}] must be a key/value pair"
                    )
                key, value = pair
                for name, cache in (("key", key), ("value", value)):
                    if not isinstance(cache, Tensor):
                        raise TypeError(
                            f"past_key_values[{layer_index}] {name} must be a torch.Tensor"
                        )
                    if (
                        cache.ndim != 4
                        or (cache.shape[0], cache.shape[1], cache.shape[3])
                        != expected_shape
                    ):
                        raise GPT2ConfigError(
                            f"past_key_values[{layer_index}] {name} has invalid cache shape"
                        )
                    if (
                        cache.dtype != self.wte.weight.dtype
                        or cache.device != token_ids.device
                    ):
                        raise GPT2ConfigError(
                            f"past_key_values[{layer_index}] {name} must match model dtype and device"
                        )
                if key.shape[2] != value.shape[2]:
                    raise GPT2ConfigError(
                        f"past_key_values[{layer_index}] key/value lengths differ"
                    )
                if layer_index == 0:
                    past_time = key.shape[2]
                elif key.shape[2] != past_time:
                    raise GPT2ConfigError("past_key_values must share past length")
        if past_time + time > base.context_length:
            raise GPT2ConfigError("token sequence exceeds context_length")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= base.vocab_size:
            raise GPT2ConfigError("token_ids contain an out-of-range id")

        hidden = self.drop(self.wte(token_ids))
        next_key_values: list[LayerKeyValue] = []
        for layer_index, block in enumerate(self.h):
            if not isinstance(block.attn, RotarySelfAttention):
                raise GPT2ConfigError("cached forward requires rotary attention")
            past_key, past_value = (
                (None, None)
                if past_key_values is None
                else past_key_values[layer_index]
            )
            attended, key, value = block.attn.forward_with_cache(
                block.ln_1(hidden), past_key=past_key, past_value=past_value
            )
            after_attention = hidden + attended
            hidden = after_attention + block.mlp(block.ln_2(after_attention))
            next_key_values.append((key, value))
        return self.lm_head(self.ln_f(hidden)), tuple(next_key_values)

    @property
    def parameter_count(self) -> int:
        """Count unique trainable values, excluding the tied output head."""

        return sum(parameter.numel() for parameter in self.parameters())
