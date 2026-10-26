"""Opt-in expert decoder preserving dense attention and dropless eval semantics."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .decoder_variant_model import DecoderVariantModel
from .moe_config import MoEConfig
from .moe_layer import MoEFeedForward


@dataclass(frozen=True)
class MoEModelOutput:
    logits: Tensor
    balance_loss: Tensor
    router_z_loss: Tensor
    routing: tuple[dict, ...]
    idle_expert_parameters: tuple[str, ...]


class MoEDecoderModel(DecoderVariantModel):
    """A CPU reference whose inherited attention and tied head remain unchanged.

    Training applies first-come capacity; eval is explicitly dropless. The
    optional mask controls routing/loss eligibility, not attention-key masking.
    Cached decoding is allowed only in eval, where capacity is chunk independent.
    """

    def __init__(self, config: MoEConfig):
        if not isinstance(config, MoEConfig):
            raise TypeError("config must be MoEConfig")
        super().__init__(config)
        for index in config.moe_layers:
            self.h[index].mlp = MoEFeedForward(config, layer_index=index)

    def forward(self, token_ids: Tensor, *, token_mask: Tensor | None = None) -> Tensor:
        return self.forward_training(token_ids, token_mask=token_mask).logits

    def forward_training(
        self, token_ids: Tensor, *, token_mask: Tensor | None = None
    ) -> MoEModelOutput:
        if (
            not isinstance(token_ids, Tensor)
            or token_ids.dtype != torch.long
            or token_ids.ndim != 2
        ):
            raise ValueError("token_ids must be rank-two torch.long")
        batch, time = token_ids.shape
        base = self.config.gpt2
        if not batch or not time or time > base.context_length:
            raise ValueError("token sequence must be nonempty and fit context_length")
        if token_ids.device != self.wte.weight.device or token_ids.device.type != "cpu":
            raise ValueError("reference MoE tokens and model must be on CPU")
        if token_ids.min() < 0 or token_ids.max() >= base.vocab_size:
            raise ValueError("token_ids contain out-of-range vocabulary IDs")
        if token_mask is not None and (
            not isinstance(token_mask, Tensor)
            or token_mask.dtype != torch.bool
            or token_mask.shape != token_ids.shape
            or token_mask.device != token_ids.device
        ):
            raise ValueError("token_mask must match boolean token dimensions")
        hidden = self.wte(token_ids)
        if self.wpe is not None:
            hidden = hidden + self.wpe(torch.arange(time, device=token_ids.device))
        hidden = self.drop(hidden)
        routing, balances, z_losses, idle = [], [], [], []
        for index, block in enumerate(self.h):
            hidden = hidden + block.attn(block.ln_1(hidden))
            normalized = block.ln_2(hidden)
            if isinstance(block.mlp, MoEFeedForward):
                result = block.mlp.forward_training(normalized, token_mask=token_mask)
                hidden = hidden + result.hidden
                balances.append(result.balance_loss)
                z_losses.append(result.router_z_loss)
                routing.append(result.stats)
                for expert in result.idle_experts:
                    idle.extend(
                        f"h.{index}.mlp.experts.{expert}.{name}"
                        for name, _ in block.mlp.experts[expert].named_parameters()
                    )
            else:
                hidden = hidden + block.mlp(normalized)
        return MoEModelOutput(
            self.lm_head(self.ln_f(hidden)),
            torch.stack(balances).mean(),
            torch.stack(z_losses).mean(),
            tuple(routing),
            tuple(idle),
        )

    def forward_with_cache(self, token_ids: Tensor, *, past_key_values=None):
        if self.training:
            raise ValueError("cached MoE decoding requires dropless eval mode")
        return super().forward_with_cache(token_ids, past_key_values=past_key_values)
