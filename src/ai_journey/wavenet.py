"""Hierarchical character-language-model primitives for curriculum Day 32."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import reduce
from hashlib import sha256
from operator import mul
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .context_mlp import ContextDataset, build_split_datasets


class WaveNetError(ValueError):
    """Raised when hierarchical model data, configuration, or state is invalid."""


@dataclass(frozen=True)
class WaveNetConfig:
    """Validated architecture for a tree-structured character language model."""

    vocab_size: int
    context_size: int = 8
    embedding_dim: int = 16
    hidden_dim: int = 64
    group_factors: tuple[int, ...] = (2, 2, 2)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_size", "embedding_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 2 if name == "vocab_size" else 1
            if value < minimum:
                raise WaveNetError(f"{name} must be at least {minimum}")

        if not isinstance(self.group_factors, tuple) or not self.group_factors:
            raise TypeError("group_factors must be a non-empty tuple of integers")
        if any(
            isinstance(factor, bool) or not isinstance(factor, int)
            for factor in self.group_factors
        ):
            raise TypeError("group_factors must contain only integers")
        if any(factor < 2 for factor in self.group_factors):
            raise WaveNetError("group_factors must be at least 2")
        if self.receptive_field != self.context_size:
            raise WaveNetError("context_size must equal the product of group_factors")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise WaveNetError("dropout must be finite and in [0, 1)")

    @property
    def receptive_field(self) -> int:
        """Return the number of input tokens consumed by the hierarchy."""

        return reduce(mul, self.group_factors, 1)

    @property
    def stage_lengths(self) -> tuple[int, ...]:
        """Return the temporal width after each consecutive grouping stage."""

        lengths: list[int] = []
        width = self.context_size
        for factor in self.group_factors:
            width //= factor
            lengths.append(width)
        return tuple(lengths)

    def fingerprint(self) -> str:
        """Return a stable digest of every architecture control."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class WaveNetTrainingConfig:
    """Validated controls for deterministic CPU optimization."""

    steps: int = 100
    batch_size: int = 32
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    seed: int = 32

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise WaveNetError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise WaveNetError(f"{name} must be positive and finite")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise WaveNetError("weight_decay must be non-negative and finite")


@dataclass(frozen=True)
class WaveNetDataset:
    """Fixed-width contexts and targets with a stable vocabulary binding."""

    vocabulary_tokens: tuple[str, ...]
    contexts: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        if len(self.vocabulary_tokens) < 2 or len(set(self.vocabulary_tokens)) != len(
            self.vocabulary_tokens
        ):
            raise WaveNetError("vocabulary must contain unique tokens")
        if any(
            not isinstance(token, str) or len(token) != 1
            for token in self.vocabulary_tokens
        ):
            raise TypeError("vocabulary tokens must be single-character strings")
        if self.contexts.ndim != 2 or self.contexts.dtype != torch.long:
            raise TypeError("contexts must be a two-dimensional torch.long tensor")
        if self.targets.ndim != 1 or self.targets.dtype != torch.long:
            raise TypeError("targets must be a one-dimensional torch.long tensor")
        if self.contexts.shape[0] != self.targets.shape[0] or not self.targets.numel():
            raise WaveNetError("contexts and targets must contain matching samples")
        if int(self.contexts.min()) < 0 or int(self.targets.min()) < 0:
            raise WaveNetError("token ids must be non-negative")
        largest = max(int(self.contexts.max()), int(self.targets.max()))
        if largest >= self.vocab_size:
            raise WaveNetError("token id is outside the vocabulary")

    @classmethod
    def from_context_dataset(cls, dataset: ContextDataset) -> WaveNetDataset:
        """Convert the established boundary-safe dataset without sharing memory."""

        if not isinstance(dataset, ContextDataset):
            raise TypeError("dataset must be ContextDataset")
        contexts = torch.from_numpy(np.array(dataset.contexts, copy=True)).long()
        targets = torch.from_numpy(np.array(dataset.targets, copy=True)).long()
        return cls(dataset.vocabulary.tokens, contexts, targets)

    @property
    def sample_count(self) -> int:
        return int(self.targets.numel())

    @property
    def context_size(self) -> int:
        return int(self.contexts.shape[1])

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary_tokens)

    def fingerprint(self) -> str:
        """Bind vocabulary order, contexts, and targets into one digest."""

        digest = sha256()
        digest.update("".join(self.vocabulary_tokens).encode())
        digest.update(self.contexts.contiguous().numpy().tobytes())
        digest.update(self.targets.contiguous().numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class WaveNetDatasetSplit:
    """Record-disjoint training and validation datasets."""

    train: WaveNetDataset
    validation: WaveNetDataset

    def __post_init__(self) -> None:
        if self.train.vocabulary_tokens != self.validation.vocabulary_tokens:
            raise WaveNetError("train and validation vocabularies must match")
        if self.train.context_size != self.validation.context_size:
            raise WaveNetError("train and validation context sizes must match")

    def fingerprint(self) -> str:
        payload = f"{self.train.fingerprint()}:{self.validation.fingerprint()}"
        return sha256(payload.encode()).hexdigest()


class WaveNetBatchCursor:
    """Deterministic shuffled batches with resumable epoch and offset."""

    def __init__(
        self, dataset: WaveNetDataset, *, batch_size: int, seed: int = 0
    ) -> None:
        if not isinstance(dataset, WaveNetDataset):
            raise TypeError("dataset must be WaveNetDataset")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise WaveNetError("batch_size must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.offset = 0

    def _order(self) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(self.dataset.sample_count, generator=generator)

    def next(self) -> tuple[Tensor, Tensor]:
        if self.offset >= self.dataset.sample_count:
            self.epoch += 1
            self.offset = 0
        indexes = self._order()[self.offset : self.offset + self.batch_size]
        self.offset += len(indexes)
        return self.dataset.contexts[indexes], self.dataset.targets[indexes]

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if not isinstance(state, dict) or set(state) != {"epoch", "offset"}:
            raise WaveNetError("batch cursor state has invalid fields")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in state.values()
        ):
            raise TypeError("batch cursor state values must be integers")
        if state["epoch"] < 0 or not 0 <= state["offset"] <= self.dataset.sample_count:
            raise WaveNetError("batch cursor state is out of range")
        self.epoch = state["epoch"]
        self.offset = state["offset"]


def build_wavenet_dataset_split(
    words: Iterable[str],
    *,
    config: WaveNetConfig,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> WaveNetDatasetSplit:
    """Build deterministic record-level splits that match an architecture."""

    if not isinstance(config, WaveNetConfig):
        raise TypeError("config must be WaveNetConfig")
    datasets = build_split_datasets(
        words,
        block_size=config.context_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    split = WaveNetDatasetSplit(
        train=WaveNetDataset.from_context_dataset(datasets.train),
        validation=WaveNetDataset.from_context_dataset(datasets.validation),
    )
    if split.train.vocab_size != config.vocab_size:
        raise WaveNetError("config vocab_size does not match the corpus vocabulary")
    return split


class FlattenConsecutive(nn.Module):
    """Concatenate adjacent time steps without changing their token order."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        if isinstance(factor, bool) or not isinstance(factor, int):
            raise TypeError("factor must be an integer")
        if factor < 2:
            raise WaveNetError("factor must be at least 2")
        self.factor = factor

    def output_shape(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Calculate the forward shape while enforcing exact grouping."""

        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) for size in shape
            )
        ):
            raise TypeError("shape must contain three integer dimensions")
        batch, time, channels = shape
        if min(shape) <= 0:
            raise WaveNetError("shape dimensions must be positive")
        if time % self.factor:
            raise WaveNetError("time dimension must be divisible by factor")
        return batch, time // self.factor, channels * self.factor

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim != 3:
            raise TypeError("inputs must be a three-dimensional tensor")
        output_shape = self.output_shape(tuple(inputs.shape))
        return inputs.reshape(output_shape)


class HierarchicalStage(nn.Module):
    """Group adjacent features, project them, normalize, and apply a nonlinearity."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        factor: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise WaveNetError(f"{name} must be positive")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise WaveNetError("dropout must be finite and in [0, 1)")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor
        self.network = nn.Sequential(
            FlattenConsecutive(factor),
            nn.Linear(factor * input_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim != 3:
            raise TypeError("inputs must be a three-dimensional tensor")
        if inputs.shape[-1] != self.input_dim:
            raise WaveNetError("input feature width does not match the stage")
        return self.network(inputs)


class HierarchicalLanguageModel(nn.Module):
    """Tree-structured character model built from a registered stage container."""

    def __init__(self, config: WaveNetConfig) -> None:
        super().__init__()
        if not isinstance(config, WaveNetConfig):
            raise TypeError("config must be WaveNetConfig")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        stages: list[HierarchicalStage] = []
        input_dim = config.embedding_dim
        for factor in config.group_factors:
            stages.append(
                HierarchicalStage(
                    input_dim,
                    config.hidden_dim,
                    factor=factor,
                    dropout=config.dropout,
                )
            )
            input_dim = config.hidden_dim
        self.stages = nn.ModuleList(stages)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 2:
            raise TypeError("token_ids must be a two-dimensional tensor")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")
        if token_ids.shape[1] != self.config.context_size:
            raise WaveNetError("token_ids must match the configured context_size")
        if token_ids.numel() and (
            int(token_ids.min()) < 0 or int(token_ids.max()) >= self.config.vocab_size
        ):
            raise WaveNetError("token id is outside the vocabulary")

        hidden = self.embedding(token_ids)
        for stage in self.stages:
            hidden = stage(hidden)
        if hidden.shape[1] != 1:
            raise RuntimeError("hierarchical grouping did not reduce time to one")
        logits = self.output(hidden[:, 0, :])

        loss = None
        if targets is not None:
            if not isinstance(targets, Tensor) or targets.shape != token_ids.shape[:1]:
                raise TypeError("targets must contain one token id per context")
            if targets.dtype != torch.long:
                raise TypeError("targets must use torch.long dtype")
            if targets.numel() and (
                int(targets.min()) < 0 or int(targets.max()) >= self.config.vocab_size
            ):
                raise WaveNetError("target token id is outside the vocabulary")
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class ShapeTraceStep:
    """One named transformation in the hierarchical model."""

    name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class WaveNetMetrics:
    """Full-dataset negative log-likelihood and derived perplexity."""

    nll: float
    perplexity: float
    sample_count: int


@dataclass(frozen=True)
class WaveNetGradientAudit:
    """Coverage and finiteness results for every registered parameter gradient."""

    parameter_tensors: int
    parameter_values: int
    missing_gradients: tuple[str, ...]
    zero_gradients: tuple[str, ...]
    nonfinite_gradients: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_gradients or self.zero_gradients or self.nonfinite_gradients
        )


@dataclass(frozen=True)
class WaveNetOverfitResult:
    """Capacity-gate result on an explicitly bounded training subset."""

    example_count: int
    steps: int
    initial_nll: float
    final_nll: float
    minimum_improvement: float
    model_fingerprint: str

    @property
    def improvement(self) -> float:
        return self.initial_nll - self.final_nll

    @property
    def passed(self) -> bool:
        return self.improvement >= self.minimum_improvement


def run_wavenet_overfit_probe(
    dataset: WaveNetDataset,
    *,
    model_config: WaveNetConfig,
    example_count: int = 8,
    steps: int = 100,
    learning_rate: float = 0.05,
    minimum_improvement: float = 0.5,
    seed: int = 32,
) -> WaveNetOverfitResult:
    """Verify that the model can reduce NLL on a declared tiny subset."""

    if not isinstance(dataset, WaveNetDataset):
        raise TypeError("dataset must be WaveNetDataset")
    if not isinstance(model_config, WaveNetConfig):
        raise TypeError("model_config must be WaveNetConfig")
    for name, value in (("example_count", example_count), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise WaveNetError(f"{name} must be positive")
    if example_count > dataset.sample_count:
        raise WaveNetError("example_count must not exceed the dataset")
    for name, value in (
        ("learning_rate", learning_rate),
        ("minimum_improvement", minimum_improvement),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise WaveNetError(f"{name} must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    subset = WaveNetDataset(
        dataset.vocabulary_tokens,
        dataset.contexts[:example_count].clone(),
        dataset.targets[:example_count].clone(),
    )
    training_config = WaveNetTrainingConfig(
        steps=steps,
        batch_size=example_count,
        learning_rate=learning_rate,
        weight_decay=0.0,
        gradient_clip=5.0,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HierarchicalLanguageModel(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.0
        )
        cursor = WaveNetBatchCursor(subset, batch_size=example_count, seed=seed)
        initial = evaluate_wavenet(model, subset)
        train_wavenet_steps(model, cursor, optimizer, training_config)
        final = evaluate_wavenet(model, subset)
    return WaveNetOverfitResult(
        example_count=example_count,
        steps=steps,
        initial_nll=initial.nll,
        final_nll=final.nll,
        minimum_improvement=minimum_improvement,
        model_fingerprint=wavenet_model_fingerprint(model),
    )


def audit_wavenet_gradients(
    model: HierarchicalLanguageModel,
    dataset: WaveNetDataset,
    *,
    example_count: int = 16,
) -> WaveNetGradientAudit:
    """Backpropagate one deterministic batch and audit the full module tree."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    if not isinstance(dataset, WaveNetDataset):
        raise TypeError("dataset must be WaveNetDataset")
    if dataset.context_size != model.config.context_size:
        raise WaveNetError("dataset context_size does not match the model")
    if dataset.vocab_size != model.config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the model")
    if isinstance(example_count, bool) or not isinstance(example_count, int):
        raise TypeError("example_count must be an integer")
    if not 1 <= example_count <= dataset.sample_count:
        raise WaveNetError("example_count must be within the dataset")

    modes = tuple((module, module.training) for module in model.modules())
    original_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    missing: list[str] = []
    zeros: list[str] = []
    nonfinite: list[str] = []
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        _, loss = model(
            dataset.contexts[:example_count], dataset.targets[:example_count]
        )
        assert loss is not None
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                missing.append(name)
            elif not bool(torch.isfinite(gradient).all()):
                nonfinite.append(name)
            elif not bool(torch.count_nonzero(gradient)):
                zeros.append(name)
    finally:
        for name, parameter in model.named_parameters():
            previous = original_gradients[name]
            parameter.grad = None if previous is None else previous
        for module, training in modes:
            module.training = training
    parameters = tuple(model.parameters())
    return WaveNetGradientAudit(
        parameter_tensors=len(parameters),
        parameter_values=sum(parameter.numel() for parameter in parameters),
        missing_gradients=tuple(missing),
        zero_gradients=tuple(zeros),
        nonfinite_gradients=tuple(nonfinite),
    )


@dataclass(frozen=True)
class WaveNetSample:
    """Generated text and the exact sampled token sequence."""

    text: str
    token_ids: tuple[int, ...]
    terminated: bool


def sample_wavenet(
    model: HierarchicalLanguageModel,
    vocabulary_tokens: tuple[str, ...],
    *,
    max_new_tokens: int = 20,
    seed: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> WaveNetSample:
    """Generate from a boundary-only context using an isolated RNG generator."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    if (
        not isinstance(vocabulary_tokens, tuple)
        or len(vocabulary_tokens) != model.config.vocab_size
        or len(set(vocabulary_tokens)) != len(vocabulary_tokens)
        or any(
            not isinstance(token, str) or len(token) != 1 for token in vocabulary_tokens
        )
    ):
        raise WaveNetError("vocabulary_tokens must match the model vocabulary")
    if vocabulary_tokens[0] != ".":
        raise WaveNetError("boundary token must be vocabulary index 0")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise WaveNetError("max_new_tokens must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise WaveNetError("temperature must be positive and finite")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise WaveNetError("top_k must be a positive integer")

    generator = torch.Generator().manual_seed(seed)
    context = torch.zeros(1, model.config.context_size, dtype=torch.long)
    sampled: list[int] = []
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(context)
                next_logits = logits[0] / temperature
                if top_k is not None:
                    values, _ = torch.topk(
                        next_logits, min(top_k, model.config.vocab_size)
                    )
                    cutoff = values[-1]
                    next_logits = next_logits.masked_fill(
                        next_logits < cutoff, float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                token_id = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
                sampled.append(token_id)
                if token_id == 0:
                    break
                context = torch.cat(
                    (context[:, 1:], torch.tensor([[token_id]], dtype=torch.long)),
                    dim=1,
                )
    finally:
        for module, training in modes:
            module.training = training
    terminated = bool(sampled and sampled[-1] == 0)
    text = "".join(vocabulary_tokens[token_id] for token_id in sampled if token_id)
    return WaveNetSample(text=text, token_ids=tuple(sampled), terminated=terminated)


@dataclass(frozen=True)
class WaveNetTrainingStep:
    """One optimization step with its pre-update loss and clipped gradient norm."""

    step: int
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class WaveNetTrainingResult:
    """Model, metrics, trace, and resumable state from one training run."""

    model: HierarchicalLanguageModel
    initial_train: WaveNetMetrics
    initial_validation: WaveNetMetrics
    final_train: WaveNetMetrics
    final_validation: WaveNetMetrics
    trace: tuple[WaveNetTrainingStep, ...]
    optimizer_state: dict[str, Any]
    cursor_state: dict[str, int]


@dataclass(frozen=True)
class WaveNetExperimentResult:
    """Self-contained, JSON-compatible evidence for one hierarchical run."""

    model_config: WaveNetConfig
    training_config: WaveNetTrainingConfig
    dataset_fingerprint: str
    model_fingerprint: str
    parameter_count: int
    completed_steps: int
    shapes: tuple[ShapeTraceStep, ...]
    initial_train: WaveNetMetrics
    initial_validation: WaveNetMetrics
    final_train: WaveNetMetrics
    final_validation: WaveNetMetrics
    trace: tuple[WaveNetTrainingStep, ...]
    sample: WaveNetSample

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        payload["report_fingerprint"] = sha256(canonical.encode()).hexdigest()
        return payload


def run_wavenet_experiment(
    datasets: WaveNetDatasetSplit,
    *,
    model_config: WaveNetConfig,
    training_config: WaveNetTrainingConfig,
    sample_seed: int = 320,
) -> WaveNetExperimentResult:
    """Train, evaluate, trace, and sample one deterministic experiment."""

    training = fit_wavenet(
        datasets,
        model_config=model_config,
        training_config=training_config,
    )
    return WaveNetExperimentResult(
        model_config=model_config,
        training_config=training_config,
        dataset_fingerprint=datasets.fingerprint(),
        model_fingerprint=wavenet_model_fingerprint(training.model),
        parameter_count=training.model.parameter_count,
        completed_steps=training_config.steps,
        shapes=trace_hierarchical_shapes(training.model),
        initial_train=training.initial_train,
        initial_validation=training.initial_validation,
        final_train=training.final_train,
        final_validation=training.final_validation,
        trace=training.trace,
        sample=sample_wavenet(
            training.model,
            datasets.train.vocabulary_tokens,
            seed=sample_seed,
        ),
    )


def verify_wavenet_report(payload: dict[str, Any]) -> None:
    """Reject incomplete or modified experiment evidence."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    expected_fields = {
        "model_config",
        "training_config",
        "dataset_fingerprint",
        "model_fingerprint",
        "parameter_count",
        "completed_steps",
        "shapes",
        "initial_train",
        "initial_validation",
        "final_train",
        "final_validation",
        "trace",
        "sample",
        "report_fingerprint",
    }
    if set(payload) != expected_fields:
        raise WaveNetError("report fields do not match the experiment schema")
    fingerprint = payload["report_fingerprint"]
    body = {key: value for key, value in payload.items() if key != "report_fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = sha256(canonical.encode()).hexdigest()
    if fingerprint != expected:
        raise WaveNetError("report fingerprint mismatch")
    completed_steps = payload["completed_steps"]
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps <= 0
        or len(payload["trace"]) != completed_steps
    ):
        raise WaveNetError("report trace does not match completed_steps")
    if payload["parameter_count"] <= 0:
        raise WaveNetError("report parameter_count must be positive")


def write_wavenet_report(path: Path, result: WaveNetExperimentResult) -> None:
    """Publish verified JSON evidence with same-directory atomic replacement."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(result, WaveNetExperimentResult):
        raise TypeError("result must be WaveNetExperimentResult")
    payload = result.to_dict()
    verify_wavenet_report(payload)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_wavenet_steps(
    model: HierarchicalLanguageModel,
    cursor: WaveNetBatchCursor,
    optimizer: torch.optim.Optimizer,
    config: WaveNetTrainingConfig,
    *,
    start_step: int = 0,
    step_count: int | None = None,
) -> tuple[WaveNetTrainingStep, ...]:
    """Train a bounded interval so runs can stop and resume exactly."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    if not isinstance(cursor, WaveNetBatchCursor):
        raise TypeError("cursor must be WaveNetBatchCursor")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(config, WaveNetTrainingConfig):
        raise TypeError("config must be WaveNetTrainingConfig")
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise TypeError("start_step must be an integer")
    if start_step < 0:
        raise WaveNetError("start_step must be non-negative")
    count = config.steps if step_count is None else step_count
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("step_count must be an integer")
    if count <= 0:
        raise WaveNetError("step_count must be positive")

    trace: list[WaveNetTrainingStep] = []
    for step in range(start_step, start_step + count):
        model.train()
        contexts, targets = cursor.next()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(contexts, targets)
        assert loss is not None
        if not torch.isfinite(loss):
            raise WaveNetError("training produced a nonfinite loss")
        loss.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        )
        if not math.isfinite(gradient_norm):
            raise WaveNetError("training produced a nonfinite gradient norm")
        optimizer.step()
        trace.append(
            WaveNetTrainingStep(
                step=step, loss=float(loss.detach()), gradient_norm=gradient_norm
            )
        )
    return tuple(trace)


def fit_wavenet(
    datasets: WaveNetDatasetSplit,
    *,
    model_config: WaveNetConfig,
    training_config: WaveNetTrainingConfig,
) -> WaveNetTrainingResult:
    """Train a hierarchical model deterministically without consuming caller RNG."""

    if not isinstance(datasets, WaveNetDatasetSplit):
        raise TypeError("datasets must be WaveNetDatasetSplit")
    if not isinstance(model_config, WaveNetConfig):
        raise TypeError("model_config must be WaveNetConfig")
    if not isinstance(training_config, WaveNetTrainingConfig):
        raise TypeError("training_config must be WaveNetTrainingConfig")
    if datasets.train.context_size != model_config.context_size:
        raise WaveNetError("dataset context_size does not match the model")
    if datasets.train.vocab_size != model_config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the model")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(training_config.seed)
        model = HierarchicalLanguageModel(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        cursor = WaveNetBatchCursor(
            datasets.train,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
        )
        initial_train = evaluate_wavenet(model, datasets.train)
        initial_validation = evaluate_wavenet(model, datasets.validation)
        trace = train_wavenet_steps(model, cursor, optimizer, training_config)

        final_train = evaluate_wavenet(model, datasets.train)
        final_validation = evaluate_wavenet(model, datasets.validation)
        return WaveNetTrainingResult(
            model=model,
            initial_train=initial_train,
            initial_validation=initial_validation,
            final_train=final_train,
            final_validation=final_validation,
            trace=trace,
            optimizer_state=copy.deepcopy(optimizer.state_dict()),
            cursor_state=cursor.state_dict(),
        )


def evaluate_wavenet(
    model: HierarchicalLanguageModel,
    dataset: WaveNetDataset,
    *,
    batch_size: int = 256,
) -> WaveNetMetrics:
    """Evaluate every sample once while preserving all module modes."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    if not isinstance(dataset, WaveNetDataset):
        raise TypeError("dataset must be WaveNetDataset")
    if dataset.context_size != model.config.context_size:
        raise WaveNetError("dataset context_size does not match the model")
    if dataset.vocab_size != model.config.vocab_size:
        raise WaveNetError("dataset vocabulary does not match the model")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise WaveNetError("batch_size must be positive")

    modes = tuple((module, module.training) for module in model.modules())
    total_loss = 0.0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, dataset.sample_count, batch_size):
                stop = min(start + batch_size, dataset.sample_count)
                _, loss = model(
                    dataset.contexts[start:stop], dataset.targets[start:stop]
                )
                assert loss is not None
                total_loss += float(loss) * (stop - start)
    finally:
        for module, training in modes:
            module.training = training
    nll = total_loss / dataset.sample_count
    return WaveNetMetrics(
        nll=nll, perplexity=math.exp(nll), sample_count=dataset.sample_count
    )


def trace_hierarchical_shapes(
    model: HierarchicalLanguageModel, *, batch_size: int = 2
) -> tuple[ShapeTraceStep, ...]:
    """Execute every registered stage and record its actual tensor contract."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise WaveNetError("batch_size must be positive")

    modes = tuple((module, module.training) for module in model.modules())
    steps: list[ShapeTraceStep] = []
    try:
        model.eval()
        with torch.no_grad():
            token_ids = torch.zeros(
                batch_size, model.config.context_size, dtype=torch.long
            )
            hidden = model.embedding(token_ids)
            steps.append(
                ShapeTraceStep(
                    "embedding",
                    tuple(token_ids.shape),
                    tuple(hidden.shape),
                    model.embedding.weight.numel(),
                )
            )
            for index, stage in enumerate(model.stages):
                inputs = hidden
                hidden = stage(hidden)
                steps.append(
                    ShapeTraceStep(
                        f"stage_{index + 1}",
                        tuple(inputs.shape),
                        tuple(hidden.shape),
                        sum(parameter.numel() for parameter in stage.parameters()),
                    )
                )
            head_inputs = hidden[:, 0, :]
            logits = model.output(head_inputs)
            steps.append(
                ShapeTraceStep(
                    "output",
                    tuple(head_inputs.shape),
                    tuple(logits.shape),
                    sum(parameter.numel() for parameter in model.output.parameters()),
                )
            )
    finally:
        for module, training in modes:
            module.training = training
    return tuple(steps)


def initialize_wavenet(
    config: WaveNetConfig, *, seed: int = 32
) -> HierarchicalLanguageModel:
    """Initialize repeatable parameters without consuming caller RNG state."""

    if not isinstance(config, WaveNetConfig):
        raise TypeError("config must be WaveNetConfig")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return HierarchicalLanguageModel(config)


WAVENET_CHECKPOINT_SCHEMA = 1


def wavenet_model_fingerprint(model: HierarchicalLanguageModel) -> str:
    """Hash all named model tensors including dtype, shape, and exact values."""

    if not isinstance(model, HierarchicalLanguageModel):
        raise TypeError("model must be HierarchicalLanguageModel")
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_finite_state(value: Any, name: str) -> None:
    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise WaveNetError(f"{name} must be finite")
    elif isinstance(value, float) and not math.isfinite(value):
        raise WaveNetError(f"{name} must be finite")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite_state(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_state(item, f"{name}[{index}]")


def save_wavenet_checkpoint(
    path: Path,
    *,
    model: HierarchicalLanguageModel,
    optimizer: torch.optim.Optimizer,
    cursor: WaveNetBatchCursor,
    training_config: WaveNetTrainingConfig,
    dataset_fingerprint: str,
    step: int,
) -> None:
    """Atomically publish every state component needed for exact continuation."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step < 0:
        raise WaveNetError("step must be non-negative")
    if (
        not isinstance(dataset_fingerprint, str)
        or len(dataset_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in dataset_fingerprint)
    ):
        raise WaveNetError("dataset_fingerprint must be a SHA-256 hex digest")
    payload = {
        "schema_version": WAVENET_CHECKPOINT_SCHEMA,
        "step": step,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "dataset_fingerprint": dataset_fingerprint,
        "model_fingerprint": wavenet_model_fingerprint(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cursor_state": cursor.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    _require_finite_state(payload["model_state"], "model_state")
    _require_finite_state(payload["optimizer_state"], "optimizer_state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_wavenet_checkpoint(
    path: Path,
    *,
    model: HierarchicalLanguageModel,
    optimizer: torch.optim.Optimizer,
    cursor: WaveNetBatchCursor,
    training_config: WaveNetTrainingConfig,
    dataset_fingerprint: str,
) -> int:
    """Validate and restore a WaveNet checkpoint transactionally."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != WAVENET_CHECKPOINT_SCHEMA:
        raise WaveNetError("unsupported checkpoint schema")
    if payload.get("model_config") != asdict(model.config):
        raise WaveNetError("checkpoint model configuration mismatch")
    if payload.get("training_config") != asdict(training_config):
        raise WaveNetError("checkpoint training configuration mismatch")
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise WaveNetError("checkpoint dataset fingerprint mismatch")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise WaveNetError("checkpoint step is invalid")
    _require_finite_state(payload.get("model_state"), "model_state")
    _require_finite_state(payload.get("optimizer_state"), "optimizer_state")

    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_cursor = cursor.state_dict()
    original_rng = torch.random.get_rng_state().clone()
    try:
        model.load_state_dict(payload["model_state"])
        if wavenet_model_fingerprint(model) != payload.get("model_fingerprint"):
            raise WaveNetError("checkpoint model fingerprint mismatch")
        optimizer.load_state_dict(payload["optimizer_state"])
        _require_finite_state(optimizer.state_dict(), "optimizer_state")
        cursor.load_state_dict(payload["cursor_state"])
        torch.random.set_rng_state(payload["torch_rng_state"])
    except Exception:
        model.load_state_dict(original_model)
        optimizer.load_state_dict(original_optimizer)
        cursor.load_state_dict(original_cursor)
        torch.random.set_rng_state(original_rng)
        raise
    return step
