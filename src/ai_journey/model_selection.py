"""Deterministic model-selection utilities for the context language model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ai_journey.bigram_lm import build_vocabulary, normalize_corpus
from ai_journey.context_mlp import (
    ContextDataset,
    ContextMLP,
    ContextMLPError,
    EvaluationMetrics,
    build_context_dataset,
    create_minibatches,
    dataset_fingerprint,
    evaluate_context_mlp,
    initialize_context_mlp,
    loss_and_gradients,
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


@dataclass(frozen=True)
class MinibatchTrainingResult:
    """Final parameters and full-training loss after every epoch."""

    model: ContextMLP
    training_nll: tuple[float, ...]


@dataclass(frozen=True)
class ValidationTrainingResult:
    """Final parameters plus train and development loss by epoch."""

    model: ContextMLP
    best_model: ContextMLP
    best_epoch: int
    training_nll: tuple[float, ...]
    development_nll: tuple[float, ...]


@dataclass(frozen=True)
class PartitionMetrics:
    """Comparable loss metrics and generalization gaps for one model."""

    train: EvaluationMetrics
    development: EvaluationMetrics
    test: EvaluationMetrics

    @property
    def development_gap(self) -> float:
        return self.development.nll - self.train.nll

    @property
    def test_gap(self) -> float:
        return self.test.nll - self.train.nll


def learning_rate_grid(
    minimum: float, maximum: float, *, count: int
) -> tuple[float, ...]:
    """Build an inclusive logarithmic learning-rate search grid."""

    for name, value in (("minimum", minimum), ("maximum", maximum)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ContextMLPError(f"{name} must be finite and positive")
    if maximum < minimum:
        raise ContextMLPError("maximum must not be below minimum")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 2:
        raise ContextMLPError("count must be at least two")
    return tuple(float(value) for value in np.geomspace(minimum, maximum, count))


def apply_sgd(
    model: ContextMLP, gradients: ContextMLP, *, learning_rate: float
) -> ContextMLP:
    """Apply one immutable stochastic-gradient update."""

    if not isinstance(model, ContextMLP) or not isinstance(gradients, ContextMLP):
        raise TypeError("model and gradients must be ContextMLP")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ContextMLPError("learning_rate must be finite and positive")
    updated: dict[str, np.ndarray] = {}
    for name, values in model.__dict__.items():
        gradient = getattr(gradients, name)
        if values.shape != gradient.shape:
            raise ContextMLPError(f"gradient shape mismatch for {name}")
        updated[name] = values - float(learning_rate) * gradient
    return ContextMLP(**updated)


def train_minibatch_context_mlp(
    dataset: ContextDataset, config: TrainingConfig
) -> MinibatchTrainingResult:
    """Train the context model using seeded shuffled minibatches."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    model = initialize_context_mlp(
        dataset,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    losses: list[float] = []
    for epoch in range(config.epochs):
        model = _train_epoch(dataset, model, config=config, epoch=epoch)
        losses.append(evaluate_context_mlp(dataset, model).nll)
    return MinibatchTrainingResult(model=model, training_nll=tuple(losses))


def train_and_validate_context_mlp(
    datasets: DatasetPartitions, config: TrainingConfig
) -> ValidationTrainingResult:
    """Evaluate development loss after every deterministic training epoch."""

    if not isinstance(datasets, DatasetPartitions):
        raise TypeError("datasets must be DatasetPartitions")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")
    model = initialize_context_mlp(
        datasets.train,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    training_nll: list[float] = []
    development_nll: list[float] = []
    best_model = model
    best_epoch = -1
    best_development_nll = float("inf")
    for epoch in range(config.epochs):
        model = _train_epoch(datasets.train, model, config=config, epoch=epoch)
        training_nll.append(evaluate_context_mlp(datasets.train, model).nll)
        current_development_nll = evaluate_context_mlp(
            datasets.development, model
        ).nll
        development_nll.append(current_development_nll)
        if current_development_nll < best_development_nll:
            best_development_nll = current_development_nll
            best_model = model
            best_epoch = epoch
    return ValidationTrainingResult(
        model=model,
        best_model=best_model,
        best_epoch=best_epoch,
        training_nll=tuple(training_nll),
        development_nll=tuple(development_nll),
    )


def evaluate_partitions(
    datasets: DatasetPartitions, model: ContextMLP
) -> PartitionMetrics:
    """Evaluate one frozen model on train, development, and test data."""

    if not isinstance(datasets, DatasetPartitions):
        raise TypeError("datasets must be DatasetPartitions")
    return PartitionMetrics(
        train=evaluate_context_mlp(datasets.train, model),
        development=evaluate_context_mlp(datasets.development, model),
        test=evaluate_context_mlp(datasets.test, model),
    )


def _train_epoch(
    dataset: ContextDataset,
    model: ContextMLP,
    *,
    config: TrainingConfig,
    epoch: int,
) -> ContextMLP:
    for batch in create_minibatches(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed + epoch,
    ):
        _, gradients = loss_and_gradients(batch, model)
        model = apply_sgd(model, gradients, learning_rate=config.learning_rate)
    return model


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
