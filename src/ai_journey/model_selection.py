"""Deterministic model-selection utilities for the context language model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ai_journey.bigram_lm import build_vocabulary, normalize_corpus
from ai_journey.context_mlp import (
    ContextDataset,
    ContextMLPError,
    build_context_dataset,
    create_minibatches,
    dataset_fingerprint,
)


@dataclass(frozen=True)
class CorpusPartitions:
    """Whole-record train, development, and test partitions."""

    train: tuple[str, ...]
    development: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class DatasetPartitions:
    """Encoded partitions that share one vocabulary and context width."""

    train: ContextDataset
    development: ContextDataset
    test: ContextDataset


@dataclass(frozen=True)
class PartitionFingerprints:
    """Content identities for each encoded evaluation partition."""

    train: str
    development: str
    test: str


@dataclass(frozen=True)
class EpochBatch:
    """One reproducibly shuffled minibatch with its training position."""

    epoch: int
    index: int
    dataset: ContextDataset


@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameters for deterministic minibatch training."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.1
    embedding_dim: int = 8
    hidden_dim: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "embedding_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ContextMLPError(f"{name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ContextMLPError("learning_rate must be finite and positive")


def split_train_dev_test(
    words: Iterable[str],
    *,
    development_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> CorpusPartitions:
    """Partition whole records deterministically without sample leakage."""

    corpus = normalize_corpus(words)
    if len(corpus) < 3:
        raise ContextMLPError("at least three records are required")
    for name, fraction in (
        ("development_fraction", development_fraction),
        ("test_fraction", test_fraction),
    ):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1
        ):
            raise ContextMLPError(f"{name} must be between zero and one")
    if development_fraction + test_fraction >= 1:
        raise ContextMLPError("development and test fractions must sum below one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    record_count = len(corpus)
    development_count = max(1, round(record_count * development_fraction))
    test_count = max(1, round(record_count * test_fraction))
    if development_count + test_count >= record_count:
        raise ContextMLPError("split fractions leave no training records")

    order = np.random.default_rng(seed).permutation(record_count)
    development_indexes = set(order[:development_count])
    test_indexes = set(order[development_count : development_count + test_count])
    return CorpusPartitions(
        train=tuple(
            word
            for index, word in enumerate(corpus)
            if index not in development_indexes | test_indexes
        ),
        development=tuple(
            word for index, word in enumerate(corpus) if index in development_indexes
        ),
        test=tuple(word for index, word in enumerate(corpus) if index in test_indexes),
    )


def build_partitioned_datasets(
    words: Iterable[str],
    *,
    block_size: int = 3,
    development_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> DatasetPartitions:
    """Encode three record partitions with a corpus-wide vocabulary."""

    corpus = normalize_corpus(words)
    vocabulary = build_vocabulary(corpus)
    partitions = split_train_dev_test(
        corpus,
        development_fraction=development_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return DatasetPartitions(
        train=build_context_dataset(
            partitions.train, block_size=block_size, vocabulary=vocabulary
        ),
        development=build_context_dataset(
            partitions.development, block_size=block_size, vocabulary=vocabulary
        ),
        test=build_context_dataset(
            partitions.test, block_size=block_size, vocabulary=vocabulary
        ),
    )


def partition_fingerprints(datasets: DatasetPartitions) -> PartitionFingerprints:
    """Fingerprint every partition so an experiment can prove its inputs."""

    if not isinstance(datasets, DatasetPartitions):
        raise TypeError("datasets must be DatasetPartitions")
    return PartitionFingerprints(
        train=dataset_fingerprint(datasets.train),
        development=dataset_fingerprint(datasets.development),
        test=dataset_fingerprint(datasets.test),
    )


def minibatch_epochs(
    dataset: ContextDataset, *, batch_size: int, epochs: int, seed: int = 0
) -> tuple[EpochBatch, ...]:
    """Return deterministic epoch-specific minibatches covering all samples."""

    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be an integer")
    if epochs <= 0:
        raise ContextMLPError("epochs must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return tuple(
        EpochBatch(epoch=epoch, index=index, dataset=batch)
        for epoch in range(epochs)
        for index, batch in enumerate(
            create_minibatches(dataset, batch_size=batch_size, seed=seed + epoch)
        )
    )
