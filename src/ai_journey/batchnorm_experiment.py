"""Controlled evidence for the BatchNorm train/eval mode trap."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import functional as F

from .transformer_lab import DecoderLanguageModel, TransformerLabError


@torch.no_grad()
def evaluate_mode_nll(
    model: DecoderLanguageModel,
    tokens: Tensor,
    *,
    training_mode: bool,
    batch_size: int = 64,
) -> float:
    """Evaluate a cloned model in an explicit mode without mutating the caller."""

    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise TypeError("tokens must be a one-dimensional torch.long tensor")
    if len(tokens) <= model.config.block_size:
        raise TransformerLabError("evaluation stream is too short")
    if not isinstance(training_mode, bool):
        raise TypeError("training_mode must be a boolean")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise TransformerLabError("batch_size must be a positive integer")

    evaluated = deepcopy(model).train(training_mode)
    losses: list[Tensor] = []
    for start in range(0, len(tokens) - model.config.block_size, batch_size):
        indexes = range(
            start,
            min(start + batch_size, len(tokens) - model.config.block_size),
        )
        inputs = torch.stack(
            [tokens[index : index + model.config.block_size] for index in indexes]
        )
        targets = torch.stack(
            [
                tokens[index + 1 : index + model.config.block_size + 1]
                for index in indexes
            ]
        )
        logits, _ = evaluated(inputs)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
                reduction="none",
            )
        )
    result = float(torch.cat(losses).mean())
    if not math.isfinite(result):
        raise TransformerLabError("mode evaluation produced non-finite NLL")
    return result
