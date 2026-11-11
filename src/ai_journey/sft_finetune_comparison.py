"""Matched pre/post fine-tuning comparison over one held-out slice."""

from __future__ import annotations

from .sft_finetune_contract import SFTFineTuneConfig
from .sft_finetune_dataset import FineTuneExample
from .sft_finetune_generation import greedy_generation
from .sft_finetune_score import score_windows
from .sft_finetune_validation import validation_evidence, validation_windows
from .transformer_lab import DecoderLanguageModel


def compare_finetune_models(
    before: DecoderLanguageModel,
    after: DecoderLanguageModel,
    validation: tuple[FineTuneExample, ...],
    config: SFTFineTuneConfig,
) -> dict:
    windows = validation_windows(
        validation,
        block_size=config.block_size,
        example_limit=config.evaluation_examples,
    )
    before_score = score_windows(before, windows)
    after_score = score_windows(after, windows)
    examples = validation[: config.evaluation_examples]
    return {
        "coverage": validation_evidence(windows),
        "before": {
            "score": before_score,
            "generations": [
                greedy_generation(before, row, new_tokens=config.generation_tokens)
                for row in examples
            ],
        },
        "after": {
            "score": after_score,
            "generations": [
                greedy_generation(after, row, new_tokens=config.generation_tokens)
                for row in examples
            ],
        },
        "delta_mean_nll": after_score["mean_nll"] - before_score["mean_nll"],
        "changed_generations": sum(
            left["token_ids"] != right["token_ids"]
            for left, right in zip(
                [
                    greedy_generation(before, row, new_tokens=config.generation_tokens)
                    for row in examples
                ],
                [
                    greedy_generation(after, row, new_tokens=config.generation_tokens)
                    for row in examples
                ],
                strict=True,
            )
        ),
    }
