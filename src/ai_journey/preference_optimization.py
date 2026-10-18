"""Length-normalized direct preference optimization for the tiny CPU decoder."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .preference_data import (
    PreferenceDataError,
    PreferencePair,
    candidate_targets,
)
from .transformer_lab import CharacterCodec, DecoderLanguageModel


def _finite_cpu_decoder(model):
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    if any(
        parameter.device.type != "cpu"
        or not parameter.is_floating_point()
        or not bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    ):
        raise PreferenceDataError(
            "preference scoring requires finite floating-point CPU parameters"
        )


def candidate_mean_logprob(
    model: DecoderLanguageModel,
    pair: PreferencePair,
    label: str,
    codec: CharacterCodec,
) -> torch.Tensor:
    """Return differentiable mean response log probability for one branch."""
    _finite_cpu_decoder(model)
    if not isinstance(codec, CharacterCodec):
        raise TypeError("codec must be CharacterCodec")
    if len(codec.tokens) != model.config.vocab_size:
        raise PreferenceDataError("codec vocabulary must match the decoder")
    targets = candidate_targets(pair, label, codec, block_size=model.config.block_size)
    values = []
    for width in range(1, model.config.block_size + 1):
        selected = [
            index
            for index, context in enumerate(targets.contexts)
            if len(context) == width
        ]
        if not selected:
            continue
        inputs = torch.tensor(
            [targets.contexts[index] for index in selected], dtype=torch.long
        )
        labels = torch.tensor(
            [targets.targets[index] for index in selected], dtype=torch.long
        )
        logits, _ = model(inputs)
        values.append(
            F.log_softmax(logits[:, -1], dim=-1).gather(1, labels[:, None]).squeeze(1)
        )
    if not values:
        raise PreferenceDataError("candidate has no scoreable response targets")
    score = torch.cat(values).mean()
    if not bool(torch.isfinite(score)):
        raise PreferenceDataError("candidate mean log probability must be finite")
    return score


def evaluate_candidate_logprob(model, pair, label, codec) -> float:
    """Score a branch without gradients and restore every caller module mode."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            value = float(candidate_mean_logprob(model, pair, label, codec))
    finally:
        for module, mode in modes:
            module.training = mode
    if not math.isfinite(value):
        raise PreferenceDataError("evaluated candidate score must be finite")
    return value
