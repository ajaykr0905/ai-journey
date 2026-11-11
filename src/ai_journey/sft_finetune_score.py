"""Exact assistant-only likelihood metrics over declared SFT windows."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .sft_finetune_batch import collate_windows
from .sft_finetune_contract import SFTFineTuneError
from .sft_finetune_windows import FineTuneWindow
from .transformer_lab import DecoderLanguageModel


@torch.no_grad()
def score_windows(
    model: DecoderLanguageModel, windows: tuple[FineTuneWindow, ...]
) -> dict[str, float | int]:
    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be DecoderLanguageModel")
    batch = collate_windows(windows)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        logits, _ = model(batch.inputs)
        selected_logits = logits[batch.target_mask]
        selected_targets = batch.targets[batch.target_mask]
        losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
        if not bool(torch.isfinite(losses).all()):
            raise SFTFineTuneError("assistant likelihood must remain finite")
        nll_sum = math.fsum(float(value) for value in losses)
        target_tokens = int(selected_targets.numel())
        mean_nll = nll_sum / target_tokens
        return {
            "nll_sum": nll_sum,
            "target_tokens": target_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
        }
    finally:
        for module, training in modes:
            module.training = training
