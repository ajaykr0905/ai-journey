"""Deterministic tiny decoder construction for bounded SFT evidence."""

from __future__ import annotations

import torch

from .sft_finetune_contract import SFTFineTuneConfig
from .transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
    seed_everything,
)


def build_finetune_model(config: SFTFineTuneConfig) -> DecoderLanguageModel:
    if not isinstance(config, SFTFineTuneConfig):
        raise TypeError("config must be SFTFineTuneConfig")
    seed_everything(config.seed)
    model = DecoderLanguageModel(
        TransformerConfig(
            vocab_size=256,
            block_size=config.block_size,
            embedding_dim=config.embedding_dim,
            head_count=config.head_count,
            layer_count=config.layer_count,
            feed_forward_expansion=2,
            dropout=0.0,
        )
    ).to(dtype=torch.float64)
    return model


def finetune_model_evidence(model: DecoderLanguageModel) -> dict:
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    return {
        "architecture_sha256": model.config.fingerprint(),
        "model_sha256": model_fingerprint(model),
        "parameters": model.parameter_count,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
