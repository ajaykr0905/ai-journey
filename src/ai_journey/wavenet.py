"""Hierarchical character-language-model primitives for curriculum Day 32."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import reduce
from hashlib import sha256
from operator import mul

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
