"""Boundary-safe context windows and a deterministic character MLP."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from ai_journey.bigram_lm import (
    BOUNDARY_TOKEN,
    Vocabulary,
    build_vocabulary,
    normalize_corpus,
)


class ContextMLPError(ValueError):
    """Raised when dataset, model, or training inputs are invalid."""


@dataclass(frozen=True)
class ContextDataset:
    vocabulary: Vocabulary
    contexts: np.ndarray
    targets: np.ndarray
    block_size: int

    @property
    def sample_count(self) -> int:
        return int(self.targets.size)


@dataclass(frozen=True)
class ContextMLP:
    embeddings: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    model: ContextMLP
    losses: tuple[float, ...]


@dataclass(frozen=True)
class CorpusSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]


@dataclass(frozen=True)
class _ForwardPass:
    flattened: np.ndarray
    hidden: np.ndarray
    probabilities: np.ndarray
    log_probabilities: np.ndarray


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ContextMLPError(f"{name} must be positive")
    return value


def split_records(
    words: Iterable[str], *, validation_fraction: float = 0.2, seed: int = 0
) -> CorpusSplit:
    """Split whole records deterministically to prevent sample leakage."""

    corpus = normalize_corpus(words)
    if len(corpus) < 2:
        raise ContextMLPError("at least two records are required for a split")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0 < validation_fraction < 1
    ):
        raise ContextMLPError("validation_fraction must be between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    order = np.random.default_rng(seed).permutation(len(corpus))
    validation_size = max(
        1, min(len(corpus) - 1, round(len(corpus) * validation_fraction))
    )
    validation_indexes = set(order[:validation_size])
    return CorpusSplit(
        train=tuple(
            word for index, word in enumerate(corpus) if index not in validation_indexes
        ),
        validation=tuple(
            word for index, word in enumerate(corpus) if index in validation_indexes
        ),
    )


def build_context_dataset(
    words: Iterable[str], *, block_size: int = 3
) -> ContextDataset:
    """Create fixed-width contexts without crossing record boundaries."""

    corpus = normalize_corpus(words)
    width = _positive_int("block_size", block_size)
    vocabulary = build_vocabulary(corpus)
    boundary_id = vocabulary.encode(BOUNDARY_TOKEN)
    contexts: list[tuple[int, ...]] = []
    targets: list[int] = []

    for word in corpus:
        context = [boundary_id] * width
        for token in (*word, BOUNDARY_TOKEN):
            target = vocabulary.encode(token)
            contexts.append(tuple(context))
            targets.append(target)
            context = [*context[1:], target]

    dataset = ContextDataset(
        vocabulary=vocabulary,
        contexts=np.asarray(contexts, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        block_size=width,
    )
    _validate_dataset(dataset)
    return dataset


def _validate_dataset(dataset: ContextDataset) -> None:
    if not isinstance(dataset, ContextDataset):
        raise TypeError("dataset must be ContextDataset")
    if not isinstance(dataset.vocabulary, Vocabulary):
        raise TypeError("vocabulary must be Vocabulary")
    _positive_int("block_size", dataset.block_size)
    if not isinstance(dataset.contexts, np.ndarray):
        raise TypeError("contexts must be a NumPy array")
    if not isinstance(dataset.targets, np.ndarray):
        raise TypeError("targets must be a NumPy array")
    expected = (dataset.targets.size, dataset.block_size)
    if dataset.contexts.shape != expected:
        raise ContextMLPError(
            f"contexts must have shape {expected}, got {dataset.contexts.shape}"
        )
    if dataset.contexts.dtype.kind not in {"i", "u"}:
        raise TypeError("contexts must use an integer dtype")
    if dataset.targets.ndim != 1 or dataset.targets.dtype.kind not in {"i", "u"}:
        raise TypeError("targets must be a one-dimensional integer array")
    if dataset.targets.size == 0:
        raise ContextMLPError("dataset must contain at least one sample")
    if np.any(dataset.contexts < 0) or np.any(
        dataset.contexts >= dataset.vocabulary.size
    ):
        raise ContextMLPError("context token is outside the vocabulary")
    if np.any(dataset.targets < 0) or np.any(
        dataset.targets >= dataset.vocabulary.size
    ):
        raise ContextMLPError("target token is outside the vocabulary")


def initialize_context_mlp(
    dataset: ContextDataset,
    *,
    embedding_dim: int = 8,
    hidden_dim: int = 64,
    seed: int = 0,
) -> ContextMLP:
    """Initialize a context MLP with deterministic Xavier-scaled weights."""

    _validate_dataset(dataset)
    embedding_dim = _positive_int("embedding_dim", embedding_dim)
    hidden_dim = _positive_int("hidden_dim", hidden_dim)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = np.random.default_rng(seed)
    input_size = dataset.block_size * embedding_dim
    vocabulary_size = dataset.vocabulary.size
    model = ContextMLP(
        embeddings=rng.normal(0.0, 0.1, (vocabulary_size, embedding_dim)),
        input_weights=rng.normal(
            0.0, sqrt(2.0 / (input_size + hidden_dim)), (input_size, hidden_dim)
        ),
        input_bias=np.zeros(hidden_dim, dtype=np.float64),
        output_weights=rng.normal(
            0.0,
            sqrt(2.0 / (hidden_dim + vocabulary_size)),
            (hidden_dim, vocabulary_size),
        ),
        output_bias=np.zeros(vocabulary_size, dtype=np.float64),
    )
    _validate_model(model, dataset)
    return model


def _validate_model(model: ContextMLP, dataset: ContextDataset) -> None:
    if not isinstance(model, ContextMLP):
        raise TypeError("model must be ContextMLP")
    for name, values in model.__dict__.items():
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if values.dtype.kind != "f":
            raise TypeError(f"{name} must use a floating-point dtype")
        if not np.all(np.isfinite(values)):
            raise ContextMLPError(f"{name} contains a non-finite value")

    vocabulary_size = dataset.vocabulary.size
    if model.embeddings.ndim != 2 or model.embeddings.shape[0] != vocabulary_size:
        raise ContextMLPError("embedding rows must match the vocabulary")
    embedding_dim = model.embeddings.shape[1]
    if model.input_weights.ndim != 2 or model.input_weights.shape[0] != (
        dataset.block_size * embedding_dim
    ):
        raise ContextMLPError("input weights do not match the flattened context")
    hidden_dim = model.input_weights.shape[1]
    expected = {
        "input_bias": (hidden_dim,),
        "output_weights": (hidden_dim, vocabulary_size),
        "output_bias": (vocabulary_size,),
    }
    for name, shape in expected.items():
        if getattr(model, name).shape != shape:
            raise ContextMLPError(f"{name} must have shape {shape}")


def predict_probabilities(dataset: ContextDataset, model: ContextMLP) -> np.ndarray:
    """Return next-token probabilities for every context."""

    return _forward(dataset, model).probabilities


def _forward(dataset: ContextDataset, model: ContextMLP) -> _ForwardPass:
    _validate_dataset(dataset)
    _validate_model(model, dataset)
    embedded = model.embeddings[dataset.contexts]
    flattened = embedded.reshape(dataset.sample_count, -1)
    hidden = np.tanh(flattened @ model.input_weights + model.input_bias)
    logits = hidden @ model.output_weights + model.output_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    log_probabilities = shifted - np.log(exponentials.sum(axis=1, keepdims=True))
    return _ForwardPass(
        flattened=flattened,
        hidden=hidden,
        probabilities=np.exp(log_probabilities),
        log_probabilities=log_probabilities,
    )


def loss_and_gradients(
    dataset: ContextDataset, model: ContextMLP
) -> tuple[float, ContextMLP]:
    """Compute mean NLL and exact full-batch gradients."""

    forward = _forward(dataset, model)
    rows = np.arange(dataset.sample_count)
    loss = float(-forward.log_probabilities[rows, dataset.targets].mean())

    logit_gradient = forward.probabilities.copy()
    logit_gradient[rows, dataset.targets] -= 1.0
    logit_gradient /= dataset.sample_count
    output_weights = forward.hidden.T @ logit_gradient
    output_bias = logit_gradient.sum(axis=0)
    hidden_gradient = logit_gradient @ model.output_weights.T
    activation_gradient = hidden_gradient * (1.0 - forward.hidden**2)
    input_weights = forward.flattened.T @ activation_gradient
    input_bias = activation_gradient.sum(axis=0)
    context_gradient = (activation_gradient @ model.input_weights.T).reshape(
        dataset.sample_count, dataset.block_size, model.embeddings.shape[1]
    )
    embeddings = np.zeros_like(model.embeddings)
    np.add.at(embeddings, dataset.contexts, context_gradient)

    return loss, ContextMLP(
        embeddings=embeddings,
        input_weights=input_weights,
        input_bias=input_bias,
        output_weights=output_weights,
        output_bias=output_bias,
    )


def train_context_mlp(
    dataset: ContextDataset,
    *,
    embedding_dim: int = 8,
    hidden_dim: int = 64,
    steps: int = 200,
    learning_rate: float = 0.1,
    seed: int = 0,
) -> TrainingResult:
    """Train with deterministic full-batch gradient descent."""

    steps = _positive_int("steps", steps)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number")
    learning_rate = float(learning_rate)
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ContextMLPError("learning_rate must be finite and positive")

    model = initialize_context_mlp(
        dataset,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    losses = []
    for _ in range(steps):
        loss, gradients = loss_and_gradients(dataset, model)
        losses.append(loss)
        model = ContextMLP(
            embeddings=model.embeddings - learning_rate * gradients.embeddings,
            input_weights=model.input_weights - learning_rate * gradients.input_weights,
            input_bias=model.input_bias - learning_rate * gradients.input_bias,
            output_weights=model.output_weights
            - learning_rate * gradients.output_weights,
            output_bias=model.output_bias - learning_rate * gradients.output_bias,
        )
    final_loss, _ = loss_and_gradients(dataset, model)
    losses.append(final_loss)
    return TrainingResult(model=model, losses=tuple(losses))
