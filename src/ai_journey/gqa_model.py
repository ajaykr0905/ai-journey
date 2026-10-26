"""Grouped-query attention composed with existing dense and expert decoders."""

import torch
from torch import nn

from .decoder_variant_model import DecoderVariantModel
from .gqa_config import GQAConfig
from .grouped_rotary_attention import GroupedRotarySelfAttention
from .moe_config import MoEConfig
from .moe_model import MoEDecoderModel, MoEModelOutput
from .gqa_resources import require_gqa_execution_budget


class GroupedQueryModel(nn.Module):
    """Explicit new state namespace and fingerprint; old models remain unchanged."""

    def __init__(self, config: GQAConfig):
        super().__init__()
        if not isinstance(config, GQAConfig):
            raise TypeError("config must be GQAConfig")
        self.config = config
        self.decoder = (
            MoEDecoderModel(config.decoder)
            if isinstance(config.decoder, MoEConfig)
            else DecoderVariantModel(config.decoder)
        )
        for block in self.decoder.h:
            block.attn = GroupedRotarySelfAttention(
                config.gpt2, kv_heads=config.kv_heads, base=config.rope_base
            )

    @property
    def h(self):
        return self.decoder.h

    @property
    def wte(self):
        return self.decoder.wte

    @property
    def wpe(self):
        return self.decoder.wpe

    @property
    def drop(self):
        return self.decoder.drop

    @property
    def ln_f(self):
        return self.decoder.ln_f

    @property
    def lm_head(self):
        return self.decoder.lm_head

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, token_ids, *, token_mask=None):
        return self.forward_training(token_ids, token_mask=token_mask).logits

    def forward_training(self, token_ids, *, token_mask=None):
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.dtype != torch.long
            or token_ids.ndim != 2
            or not token_ids.numel()
        ):
            raise ValueError("grouped decoder requires nonempty long token matrices")
        require_gqa_execution_budget(
            self.config,
            batch_size=token_ids.shape[0],
            sequence_length=token_ids.shape[1],
            dtype=str(self.wte.weight.dtype).removeprefix("torch."),
        )
        if token_mask is not None and (
            not isinstance(token_mask, torch.Tensor)
            or token_mask.dtype != torch.bool
            or token_mask.shape != token_ids.shape
            or token_mask.device != token_ids.device
        ):
            raise ValueError("token_mask must match boolean token dimensions")
        if isinstance(self.decoder, MoEDecoderModel):
            result = self.decoder.forward_training(token_ids, token_mask=token_mask)
            return MoEModelOutput(
                result.logits,
                result.balance_loss,
                result.router_z_loss,
                result.routing,
                tuple(f"decoder.{name}" for name in result.idle_expert_parameters),
            )
        logits = self.decoder(token_ids)
        zero = logits.sum() * 0
        return MoEModelOutput(logits, zero, zero.double(), (), ())

    def forward_with_cache(self, token_ids, *, past_key_values=None):
        """Advance native KV heads at absolute RoPE offsets without mutation."""
        if any(module.training for module in self.modules()):
            raise ValueError(
                "cached grouped decoder requires every module in eval mode"
            )
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.dtype != torch.long
            or token_ids.ndim != 2
            or not token_ids.numel()
            or token_ids.device != self.wte.weight.device
            or token_ids.device.type != "cpu"
        ):
            raise ValueError(
                "cached grouped decoder requires nonempty CPU long matrices"
            )
        batch, time = token_ids.shape
        if token_ids.min() < 0 or token_ids.max() >= self.config.gpt2.vocab_size:
            raise ValueError("cached grouped decoder token IDs exceed vocabulary")
        offset = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple) or len(past_key_values) != len(
                self.h
            ):
                raise ValueError("compact cache must have one pair per layer")
            for index, pair in enumerate(past_key_values):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("compact layer cache must be a key/value pair")
                for tensor in pair:
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.ndim != 4
                        or tensor.shape[:2] != (batch, self.config.kv_heads)
                        or tensor.shape[-1] != self.config.gpt2.head_dim
                        or tensor.shape[2] > self.config.gpt2.context_length
                        or tensor.dtype != self.wte.weight.dtype
                        or tensor.device != token_ids.device
                    ):
                        raise ValueError(
                            "compact cache batch/KV-heads/dtype/device must match"
                        )
                if pair[0].shape != pair[1].shape:
                    raise ValueError("compact key/value lengths must match")
                if index == 0:
                    offset = pair[0].shape[2]
                elif offset != pair[0].shape[2]:
                    raise ValueError("compact cache layers must share a prefix length")
        if offset + time > self.config.gpt2.context_length:
            raise ValueError("cached grouped decoder exceeds context_length")
        require_gqa_execution_budget(
            self.config,
            batch_size=batch,
            sequence_length=offset + time,
            query_tokens=time,
            dtype=str(self.wte.weight.dtype).removeprefix("torch."),
        )
        if past_key_values is not None and any(
            not bool(torch.isfinite(tensor).all())
            for pair in past_key_values
            for tensor in pair
        ):
            raise ValueError("compact cache values must be finite")
        hidden = self.drop(self.wte(token_ids))
        pairs = []
        for index, block in enumerate(self.h):
            key, value = (
                (None, None) if past_key_values is None else past_key_values[index]
            )
            attended, key, value = block.attn.forward_with_cache(
                block.ln_1(hidden), past_key=key, past_value=value
            )
            hidden = hidden + attended
            hidden = hidden + block.mlp(block.ln_2(hidden))
            pairs.append((key, value))
        return self.lm_head(self.ln_f(hidden)), tuple(pairs)
