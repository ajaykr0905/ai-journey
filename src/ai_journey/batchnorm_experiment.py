"""Controlled evidence for the BatchNorm train/eval mode trap."""

from __future__ import annotations

import json
import math
import platform
import random
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
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

BATCHNORM_SCHEMA_VERSION = 1


@contextmanager
def _preserve_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic)


@dataclass(frozen=True)
class BatchNormModeTrap:
    """Correct and mistaken inference measurements for one trained model."""

    eval_nll: float
    train_mode_nll: float
    train_minus_eval_nll: float


@dataclass(frozen=True)
class BatchNormRuntimeMetadata:
    """Public-safe runtime provenance for the CPU experiment."""

    python_version: str
    torch_version: str
    numpy_version: str
    device: str
    machine: str
    deterministic_algorithms: bool
    intraop_threads: int


@dataclass(frozen=True)
class BatchNormCriteria:
    """Predeclared thresholds for useful, non-leaking BatchNorm evidence."""

    min_training_loss_reduction: float = 0.01
    min_mode_nll_gap: float = 1e-4
    min_train_batch_coupling: float = 1e-4
    max_eval_batch_coupling: float = 1e-7

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise TransformerLabError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class BatchNormEvaluation:
    """Measured gate values and named violations."""

    training_loss_reduction: float
    absolute_mode_nll_gap: float
    train_batch_coupling: float
    eval_batch_coupling: float
    passed: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class BatchNormExperimentResult:
    """Deterministic transformer evidence for scratch BatchNorm behavior."""

    corpus_fingerprint: str
    model_config: TransformerConfig
    training_config: TrainingConfig
    initial_model_fingerprint: str
    first_batch_fingerprint: str
    initial_eval_nll: float
    trace: tuple[StepMetric, ...]
    final_train_nll: float
    mode_trap: BatchNormModeTrap
    batch_coupling: BatchCouplingResult
    layer_state_fingerprints: tuple[tuple[str, str], ...]
    model_fingerprint: str
    runtime: BatchNormRuntimeMetadata

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": BATCHNORM_SCHEMA_VERSION,
            "corpus_fingerprint": self.corpus_fingerprint,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "initial_model_fingerprint": self.initial_model_fingerprint,
            "first_batch_fingerprint": self.first_batch_fingerprint,
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
            "runtime": asdict(self.runtime),
        }

    def evidence_fingerprint(self) -> str:
        payload = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["evidence_fingerprint"] = self.evidence_fingerprint()
        return payload


def evaluate_batchnorm_experiment(
    result: BatchNormExperimentResult,
    criteria: BatchNormCriteria,
) -> BatchNormEvaluation:
    """Apply explicit gates to a completed BatchNorm experiment."""

    if not isinstance(result, BatchNormExperimentResult):
        raise TypeError("result must be BatchNormExperimentResult")
    if not isinstance(criteria, BatchNormCriteria):
        raise TypeError("criteria must be BatchNormCriteria")
    initial_loss = result.trace[0].loss
    final_loss = result.trace[-1].loss
    loss_reduction = (initial_loss - final_loss) / initial_loss
    mode_gap = abs(result.mode_trap.train_minus_eval_nll)
    train_coupling = result.batch_coupling.train_max_abs_delta
    eval_coupling = result.batch_coupling.eval_max_abs_delta
    violations: list[str] = []
    if loss_reduction < criteria.min_training_loss_reduction:
        violations.append("training_loss_reduction")
    if mode_gap < criteria.min_mode_nll_gap:
        violations.append("mode_nll_gap")
    if train_coupling < criteria.min_train_batch_coupling:
        violations.append("train_batch_coupling")
    if eval_coupling > criteria.max_eval_batch_coupling:
        violations.append("eval_batch_coupling")
    return BatchNormEvaluation(
        training_loss_reduction=loss_reduction,
        absolute_mode_nll_gap=mode_gap,
        train_batch_coupling=train_coupling,
        eval_batch_coupling=eval_coupling,
        passed=not violations,
        violations=tuple(violations),
    )


def build_batchnorm_report(
    result: BatchNormExperimentResult,
    criteria: BatchNormCriteria,
) -> dict[str, Any]:
    """Build a self-verifying report with inputs, results, and gate outcome."""

    evaluation = evaluate_batchnorm_experiment(result, criteria)
    payload: dict[str, Any] = {
        "experiment": result.to_dict(),
        "criteria": asdict(criteria),
        "evaluation": asdict(evaluation),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["report_fingerprint"] = sha256(encoded).hexdigest()
    return payload


def write_batchnorm_report(
    path: Path,
    result: BatchNormExperimentResult,
    criteria: BatchNormCriteria,
) -> None:
    """Atomically write a canonical JSON report."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    payload = build_batchnorm_report(result, criteria)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)


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


def _run_batchnorm_experiment(
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
    initial_model_fingerprint = model_fingerprint(model)
    initial_eval_nll = evaluate_nll(model, corpus.validation_tokens)
    control_cursor = BatchCursor(
        corpus.train_tokens,
        block_size=model_config.block_size,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    control_inputs, control_targets = control_cursor.next()
    first_batch_digest = sha256()
    for tensor in (control_inputs, control_targets):
        value = tensor.detach().cpu().contiguous()
        first_batch_digest.update(str(value.dtype).encode())
        first_batch_digest.update(str(tuple(value.shape)).encode())
        first_batch_digest.update(value.numpy().tobytes())
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
        initial_model_fingerprint=initial_model_fingerprint,
        first_batch_fingerprint=first_batch_digest.hexdigest(),
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
        runtime=BatchNormRuntimeMetadata(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            device="cpu",
            machine=platform.machine() or "unknown",
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            intraop_threads=torch.get_num_threads(),
        ),
    )


def run_batchnorm_experiment(
    corpus: TokenCorpus,
    *,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
) -> BatchNormExperimentResult:
    """Run the experiment without changing caller RNG or determinism state."""

    with _preserve_random_state():
        return _run_batchnorm_experiment(
            corpus,
            model_config=model_config,
            training_config=training_config,
        )
