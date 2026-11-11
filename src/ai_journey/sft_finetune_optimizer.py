"""Explicit AdamW parameter grouping for fine-tuning experiments."""

from __future__ import annotations

import torch

from .sft_finetune_contract import SFTFineTuneConfig
from .transformer_lab import DecoderLanguageModel


def build_finetune_optimizer(
    model: DecoderLanguageModel, config: SFTFineTuneConfig
) -> torch.optim.AdamW:
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    if not isinstance(config, SFTFineTuneConfig):
        raise TypeError("config must be SFTFineTuneConfig")
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    if len(decay) + len(no_decay) != len(tuple(model.parameters())):
        raise RuntimeError("optimizer parameter grouping is incomplete")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
