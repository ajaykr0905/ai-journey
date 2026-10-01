"""Controlled evidence for the BatchNorm train/eval mode trap."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .batch_normalization import (
    BatchCouplingResult,
    ScratchBatchNorm,
    measure_batch_coupling,
    snapshot_batch_norm,
)
from .transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    build_optimizer,
    evaluate_nll,
    model_fingerprint,
    seed_everything,
    train_steps,
)


@dataclass(frozen=True)
class BatchNormModeTrap:
    """Correct and mistaken inference measurements for one trained model."""

    eval_nll: float
    train_mode_nll: float
    train_minus_eval_nll: float


@dataclass(frozen=True)
class BatchNormExperimentResult:
    """Deterministic transformer evidence for scratch BatchNorm behavior."""

    corpus_fingerprint: str
    model_config: TransformerConfig
    training_config: TrainingConfig
    initial_eval_nll: float
    trace: tuple[StepMetric, ...]
    final_train_nll: float
    mode_trap: BatchNormModeTrap
    batch_coupling: BatchCouplingResult
    layer_state_fingerprints: tuple[tuple[str, str], ...]
    model_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "initial_eval_nll": self.initial_eval_nll,
            "trace": [asdict(metric) for metric in self.trace],
            "final_train_nll": self.final_train_nll,
            "mode_trap": asdict(self.mode_trap),
            "batch_coupling": asdict(self.batch_coupling),
            "layer_state_fingerprints": [
                {"name": name, "fingerprint": fingerprint}
                for name, fingerprint in self.layer_state_fingerprints
            ],
            "model_fingerprint": self.model_fingerprint,
        }


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


def run_batchnorm_experiment(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> BatchNormExperimentResult:
    """Train one scratch-BatchNorm transformer and expose its mode trap."""

    if not isinstance(corpus, TokenCorpus):
        raise TypeError("corpus must be TokenCorpus")
    if model_config.normalization_mode != "scratch_batch_norm":
        raise TransformerLabError(
            "model_config normalization_mode must be scratch_batch_norm"
        )
    if model_config.vocab_size != corpus.vocab_size:
        raise TransformerLabError("model vocabulary does not match corpus")
    seed_everything(training_config.seed)
    model = DecoderLanguageModel(model_config)
    initial_eval_nll = evaluate_nll(model, corpus.validation_tokens)
    cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    trace = train_steps(
        model,
        cursor,
        build_optimizer(model, training_config),
        training_config,
    )
    final_train_nll = evaluate_nll(model, corpus.train_tokens)
    eval_nll = evaluate_mode_nll(model, corpus.validation_tokens, training_mode=False)
    train_mode_nll = evaluate_mode_nll(
        model, corpus.validation_tokens, training_mode=True
    )
    block = model_config.block_size
    anchor = corpus.validation_tokens[:block].reshape(1, block)
    companion = corpus.validation_tokens[1 : block + 1].reshape(1, block)
    coupling = measure_batch_coupling(model, anchor, companion)
    states = tuple(
        (name, snapshot_batch_norm(module).fingerprint())
        for name, module in model.named_modules()
        if isinstance(module, ScratchBatchNorm)
    )
    return BatchNormExperimentResult(
        corpus_fingerprint=corpus.fingerprint(),
        model_config=model_config,
        training_config=training_config,
        initial_eval_nll=initial_eval_nll,
        trace=trace,
        final_train_nll=final_train_nll,
        mode_trap=BatchNormModeTrap(
            eval_nll=eval_nll,
            train_mode_nll=train_mode_nll,
            train_minus_eval_nll=train_mode_nll - eval_nll,
        ),
        batch_coupling=coupling,
        layer_state_fingerprints=states,
        model_fingerprint=model_fingerprint(model),
    )
