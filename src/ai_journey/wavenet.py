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
from torch import Tensor

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
