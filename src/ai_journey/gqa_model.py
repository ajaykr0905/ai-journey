"""Grouped-query attention composed with existing dense and expert decoders."""

import torch
from torch import nn

from .decoder_variant_model import DecoderVariantModel
from .gqa_config import GQAConfig
from .grouped_rotary_attention import GroupedRotarySelfAttention
from .moe_config import MoEConfig
from .moe_model import MoEDecoderModel, MoEModelOutput


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
