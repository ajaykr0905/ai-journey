"""Day 20: a deterministic neural character-bigram language model.

The model is deliberately small enough to inspect. A one-hot character vector
selects one row of a weight matrix, stable softmax turns that row into a next-
character distribution, and mean negative log-likelihood supplies the loss.
The module also shows an exact bridge to the Day 19 count model: initializing
weights with the logarithm of additively smoothed counts produces the same
probability matrix and therefore the same corpus loss.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, isfinite

import numpy as np

from ai_journey.bigram_lm import (
    DEFAULT_CORPUS,
    BigramModel,
    SampleTrace,
    Vocabulary,
    build_bigram_model,
    build_vocabulary,
    iter_bigrams,
    normalize_corpus,
    sample_words,
    validate_model,
)


class NeuralBigramValidationError(ValueError):
    """Raised when Day 20 data, math, or experiment invariants fail."""


@dataclass(frozen=True)
class BigramDataset:
    """Integer input/target ids for every boundary-aware corpus transition."""

    vocabulary: Vocabulary
    inputs: np.ndarray
    targets: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.inputs.size)


@dataclass(frozen=True)
class TrainingStep:
    """One deterministic full-batch optimization checkpoint."""

    step: int
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class GradientProbe:
    """Analytic and centered finite-difference values for one weight."""

    row: int
    column: int
    analytic: float
    numeric: float
    absolute_error: float


@dataclass(frozen=True)
class NeuralBigramExperiment:
    """All reproducible evidence produced by the Day 20 experiment."""

    dataset: BigramDataset
    count_model: BigramModel
    smoothing: float
    smoothed_probabilities: np.ndarray
    exact_weights: np.ndarray
    count_loss: float
    exact_neural_loss: float
    initial_loss: float
    trained_loss: float
    trained_weights: np.ndarray
    training_trace: tuple[TrainingStep, ...]
    gradient_probes: tuple[GradientProbe, ...]
    samples: tuple[SampleTrace, ...]
    seed: int


def build_bigram_dataset(words: Iterable[str]) -> BigramDataset:
    """Encode all boundary-aware bigrams as parallel integer arrays."""

    corpus = normalize_corpus(words)
    vocabulary = build_vocabulary(corpus)
    inputs: list[int] = []
    targets: list[int] = []
    for word in corpus:
        for previous, next_token in iter_bigrams(word):
            inputs.append(vocabulary.encode(previous))
            targets.append(vocabulary.encode(next_token))
    dataset = BigramDataset(
        vocabulary=vocabulary,
        inputs=np.asarray(inputs, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: BigramDataset) -> None:
    """Require aligned, non-empty, in-range integer token-id vectors."""

    if not isinstance(dataset, BigramDataset):
        raise TypeError("dataset must be BigramDataset")
    for name, values in (("inputs", dataset.inputs), ("targets", dataset.targets)):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if values.ndim != 1:
            raise NeuralBigramValidationError(f"{name} must be one-dimensional")
        if values.dtype.kind not in {"i", "u"}:
            raise TypeError(f"{name} must use an integer dtype")
    if dataset.inputs.size == 0:
        raise NeuralBigramValidationError("dataset must contain at least one bigram")
    if dataset.inputs.shape != dataset.targets.shape:
        raise NeuralBigramValidationError("input and target shapes must match")
    for name, values in (("inputs", dataset.inputs), ("targets", dataset.targets)):
        if np.any(values < 0) or np.any(values >= dataset.vocabulary.size):
            raise NeuralBigramValidationError(f"{name} contain an out-of-range id")


def one_hot(token_ids: np.ndarray, vocabulary_size: int) -> np.ndarray:
    """Return a float64 one-hot matrix without mutating the input ids."""

    if not isinstance(token_ids, np.ndarray):
        raise TypeError("token_ids must be a NumPy array")
    if token_ids.ndim != 1 or token_ids.dtype.kind not in {"i", "u"}:
        raise TypeError("token_ids must be a one-dimensional integer array")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        raise TypeError("vocabulary_size must be an integer")
    if vocabulary_size <= 0:
        raise NeuralBigramValidationError("vocabulary_size must be positive")
    if np.any(token_ids < 0) or np.any(token_ids >= vocabulary_size):
        raise NeuralBigramValidationError("token id is outside the vocabulary")
    return np.eye(vocabulary_size, dtype=np.float64)[token_ids]


def validate_weights(weights: np.ndarray, vocabulary_size: int) -> None:
    """Require a finite square float matrix matching the vocabulary."""

    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy array")
    expected = (vocabulary_size, vocabulary_size)
    if weights.shape != expected:
        raise NeuralBigramValidationError(
            f"weights must have shape {expected}, got {weights.shape}"
        )
    if weights.dtype.kind != "f":
        raise TypeError("weights must use a floating-point dtype")
    if not np.all(np.isfinite(weights)):
        raise NeuralBigramValidationError("weights contain a non-finite value")


def logits_from_weights(dataset: BigramDataset, weights: np.ndarray) -> np.ndarray:
    """Compute logits with the explicit one-hot-matrix multiplication."""

    validate_dataset(dataset)
    validate_weights(weights, dataset.vocabulary.size)
    encoded = one_hot(dataset.inputs, dataset.vocabulary.size)
    return encoded @ weights


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Normalize finite 2D logits after subtracting each row maximum."""

    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a NumPy array")
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise NeuralBigramValidationError("logits must be a non-empty 2D matrix")
    if not np.all(np.isfinite(logits)):
        raise NeuralBigramValidationError("logits contain a non-finite value")
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def validate_probability_matrix(probabilities: np.ndarray) -> None:
    """Require finite, strictly positive rows that sum to one."""

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a NumPy array")
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise NeuralBigramValidationError("probabilities must be a non-empty 2D matrix")
    if not np.all(np.isfinite(probabilities)):
        raise NeuralBigramValidationError("probabilities contain a non-finite value")
    if np.any(probabilities <= 0.0):
        raise NeuralBigramValidationError("probabilities must be strictly positive")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12):
        raise NeuralBigramValidationError("probability rows must sum to one")


def negative_log_likelihood(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Return mean negative log probability assigned to integer targets."""

    validate_probability_matrix(probabilities)
    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a NumPy array")
    if targets.ndim != 1 or targets.dtype.kind not in {"i", "u"}:
        raise TypeError("targets must be a one-dimensional integer array")
    if probabilities.shape[0] != targets.size:
        raise NeuralBigramValidationError(
            "probability rows and target count must match"
        )
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise NeuralBigramValidationError("target id is outside the probability matrix")
    selected = probabilities[np.arange(targets.size), targets]
    return float(-np.log(selected).mean())


def perplexity(loss: float) -> float:
    """Convert a finite non-negative mean NLL into perplexity."""

    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise TypeError("loss must be a real number")
    value = float(loss)
    if not isfinite(value) or value < 0.0:
        raise NeuralBigramValidationError("loss must be finite and non-negative")
    try:
        result = exp(value)
    except OverflowError as exc:
        raise NeuralBigramValidationError(
            "loss is too large for finite perplexity"
        ) from exc
    if not isfinite(result):
        raise NeuralBigramValidationError("perplexity must be finite")
    return result


def validate_smoothing(smoothing: float) -> float:
    """Return additive smoothing as a finite, strictly positive float."""

    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError("smoothing must be a real number")
    value = float(smoothing)
    if not isfinite(value) or value <= 0.0:
        raise NeuralBigramValidationError(
            "smoothing must be finite and strictly positive"
        )
    return value


def additive_smoothed_probabilities(
    counts: np.ndarray, smoothing: float = 1.0
) -> np.ndarray:
    """Apply additive smoothing and row-normalize a square count matrix."""

    alpha = validate_smoothing(smoothing)
    if not isinstance(counts, np.ndarray):
        raise TypeError("counts must be a NumPy array")
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise NeuralBigramValidationError("counts must be a square matrix")
    if counts.dtype.kind not in {"i", "u"}:
        raise TypeError("counts must use an integer dtype")
    if np.any(counts < 0):
        raise NeuralBigramValidationError("counts must be non-negative")
    adjusted = counts.astype(np.float64) + alpha
    probabilities = adjusted / adjusted.sum(axis=1, keepdims=True)
    validate_probability_matrix(probabilities)
    return probabilities


def log_count_weights(counts: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Map smoothed counts to logits whose softmax is the count distribution."""

    alpha = validate_smoothing(smoothing)
    probabilities = additive_smoothed_probabilities(counts, alpha)
    del probabilities  # validation above establishes the count-matrix contract
    return np.log(counts.astype(np.float64) + alpha)


def count_model_loss(dataset: BigramDataset, probabilities: np.ndarray) -> float:
    """Evaluate a vocabulary-square count probability matrix on the dataset."""

    validate_dataset(dataset)
    validate_probability_matrix(probabilities)
    expected = (dataset.vocabulary.size, dataset.vocabulary.size)
    if probabilities.shape != expected:
        raise NeuralBigramValidationError(
            f"count probabilities must have shape {expected}"
        )
    rows = probabilities[dataset.inputs]
    return negative_log_likelihood(rows, dataset.targets)


def neural_bigram_loss(dataset: BigramDataset, weights: np.ndarray) -> float:
    """Evaluate one-hot linear logits followed by softmax and mean NLL."""

    logits = logits_from_weights(dataset, weights)
    return negative_log_likelihood(stable_softmax(logits), dataset.targets)


def loss_and_gradient(
    dataset: BigramDataset, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return mean NLL and its analytic gradient with respect to weights."""

    logits = logits_from_weights(dataset, weights)
    probabilities = stable_softmax(logits)
    loss = negative_log_likelihood(probabilities, dataset.targets)
    encoded_inputs = one_hot(dataset.inputs, dataset.vocabulary.size)
    encoded_targets = one_hot(dataset.targets, dataset.vocabulary.size)
    logits_gradient = (probabilities - encoded_targets) / dataset.sample_count
    weights_gradient = encoded_inputs.T @ logits_gradient
    return loss, weights_gradient


def finite_difference_probes(
    dataset: BigramDataset,
    weights: np.ndarray,
    coordinates: Sequence[tuple[int, int]],
    *,
    epsilon: float = 1e-6,
) -> tuple[GradientProbe, ...]:
    """Compare analytic gradients with centered differences at selected weights."""

    validate_dataset(dataset)
    validate_weights(weights, dataset.vocabulary.size)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a real number")
    epsilon_value = float(epsilon)
    if not isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise NeuralBigramValidationError("epsilon must be finite and positive")
    if not coordinates:
        raise NeuralBigramValidationError(
            "at least one gradient coordinate is required"
        )

    _, analytic_gradient = loss_and_gradient(dataset, weights)
    probes: list[GradientProbe] = []
    seen: set[tuple[int, int]] = set()
    for coordinate in coordinates:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in coordinate
            )
        ):
            raise TypeError("gradient coordinates must be (row, column) integer tuples")
        row, column = coordinate
        if coordinate in seen:
            raise NeuralBigramValidationError("gradient coordinates must be unique")
        seen.add(coordinate)
        if not (0 <= row < weights.shape[0] and 0 <= column < weights.shape[1]):
            raise NeuralBigramValidationError("gradient coordinate is out of range")
        plus = weights.copy()
        minus = weights.copy()
        plus[row, column] += epsilon_value
        minus[row, column] -= epsilon_value
        numeric = (
            neural_bigram_loss(dataset, plus) - neural_bigram_loss(dataset, minus)
        ) / (2.0 * epsilon_value)
        analytic = float(analytic_gradient[row, column])
        probes.append(
            GradientProbe(
                row=row,
                column=column,
                analytic=analytic,
                numeric=float(numeric),
                absolute_error=abs(analytic - float(numeric)),
            )
        )
    return tuple(probes)


def train_neural_bigram(
    dataset: BigramDataset,
    *,
    steps: int = 200,
    learning_rate: float = 10.0,
    checkpoint_interval: int = 20,
) -> tuple[np.ndarray, tuple[TrainingStep, ...]]:
    """Train zero-initialized weights with deterministic full-batch descent."""

    validate_dataset(dataset)
    for name, value in (("steps", steps), ("checkpoint_interval", checkpoint_interval)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise NeuralBigramValidationError(f"{name} must be positive")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise TypeError("learning_rate must be a real number")
    rate = float(learning_rate)
    if not isfinite(rate) or rate <= 0.0:
        raise NeuralBigramValidationError("learning_rate must be finite and positive")

    weights = np.zeros(
        (dataset.vocabulary.size, dataset.vocabulary.size), dtype=np.float64
    )
    trace: list[TrainingStep] = []
    for step in range(steps + 1):
        loss, gradient = loss_and_gradient(dataset, weights)
        if step == 0 or step == steps or step % checkpoint_interval == 0:
            trace.append(
                TrainingStep(step, loss, float(np.linalg.norm(gradient, ord=2)))
            )
        if step < steps:
            weights = weights - rate * gradient
    return weights, tuple(trace)


def probability_model_from_weights(
    count_model: BigramModel, weights: np.ndarray
) -> BigramModel:
    """Build a sampling-compatible model from a neural weight matrix."""

    validate_model(count_model)
    validate_weights(weights, count_model.vocabulary.size)
    probabilities = stable_softmax(weights)
    model = BigramModel(count_model.vocabulary, count_model.counts, probabilities)
    validate_model(model)
    return model


def run_neural_bigram_experiment(
    words: Iterable[str] = DEFAULT_CORPUS,
    *,
    smoothing: float = 1.0,
    steps: int = 200,
    learning_rate: float = 10.0,
    sample_count: int = 10,
    seed: int = 2020,
) -> NeuralBigramExperiment:
    """Run smoothing equivalence, gradient checks, training, and sampling."""

    corpus = normalize_corpus(words)
    dataset = build_bigram_dataset(corpus)
    count_model = build_bigram_model(corpus)
    alpha = validate_smoothing(smoothing)
    smoothed = additive_smoothed_probabilities(count_model.counts, alpha)
    exact_weights = log_count_weights(count_model.counts, alpha)
    count_loss = count_model_loss(dataset, smoothed)
    exact_neural_loss = neural_bigram_loss(dataset, exact_weights)
    zero_weights = np.zeros_like(exact_weights)
    initial_loss = neural_bigram_loss(dataset, zero_weights)
    probes = finite_difference_probes(
        dataset,
        zero_weights,
        ((0, 1), (0, dataset.vocabulary.size - 1), (1, 0), (2, 3), (5, 7), (10, 4)),
    )
    trained_weights, trace = train_neural_bigram(
        dataset,
        steps=steps,
        learning_rate=learning_rate,
        checkpoint_interval=max(1, steps // 10),
    )
    trained_loss = neural_bigram_loss(dataset, trained_weights)
    sampling_model = probability_model_from_weights(count_model, trained_weights)
    experiment = NeuralBigramExperiment(
        dataset=dataset,
        count_model=count_model,
        smoothing=alpha,
        smoothed_probabilities=smoothed,
        exact_weights=exact_weights,
        count_loss=count_loss,
        exact_neural_loss=exact_neural_loss,
        initial_loss=initial_loss,
        trained_loss=trained_loss,
        trained_weights=trained_weights,
        training_trace=trace,
        gradient_probes=probes,
        samples=sample_words(sampling_model, count=sample_count, seed=seed),
        seed=seed,
    )
    validate_experiment(experiment)
    return experiment


def validate_experiment(experiment: NeuralBigramExperiment) -> None:
    """Validate equivalence, gradients, optimization, shapes, and samples."""

    if not isinstance(experiment, NeuralBigramExperiment):
        raise TypeError("experiment must be NeuralBigramExperiment")
    validate_dataset(experiment.dataset)
    validate_model(experiment.count_model)
    size = experiment.dataset.vocabulary.size
    if experiment.dataset.vocabulary != experiment.count_model.vocabulary:
        raise NeuralBigramValidationError("dataset and count model vocabulary differ")
    if experiment.dataset.sample_count != int(experiment.count_model.counts.sum()):
        raise NeuralBigramValidationError("dataset and count transition totals differ")
    validate_probability_matrix(experiment.smoothed_probabilities)
    if experiment.smoothed_probabilities.shape != (size, size):
        raise NeuralBigramValidationError("smoothed probability shape is invalid")
    validate_weights(experiment.exact_weights, size)
    validate_weights(experiment.trained_weights, size)
    exact_probabilities = stable_softmax(experiment.exact_weights)
    if not np.allclose(
        exact_probabilities, experiment.smoothed_probabilities, atol=1e-12
    ):
        raise NeuralBigramValidationError(
            "log-count neural probabilities do not match smoothed counts"
        )
    if abs(experiment.count_loss - experiment.exact_neural_loss) > 1e-12:
        raise NeuralBigramValidationError("count and exact neural losses do not match")
    if not experiment.training_trace:
        raise NeuralBigramValidationError("training trace must not be empty")
    if experiment.training_trace[0].step != 0:
        raise NeuralBigramValidationError("training trace must begin at step zero")
    if experiment.initial_loss != experiment.training_trace[0].loss:
        raise NeuralBigramValidationError("initial loss and trace do not match")
    if experiment.trained_loss != experiment.training_trace[-1].loss:
        raise NeuralBigramValidationError("trained loss and trace do not match")
    if experiment.trained_loss >= experiment.initial_loss:
        raise NeuralBigramValidationError("training did not reduce loss")
    if max(probe.absolute_error for probe in experiment.gradient_probes) > 1e-8:
        raise NeuralBigramValidationError("finite-difference gradient check failed")
    if not experiment.samples:
        raise NeuralBigramValidationError("experiment must include samples")
    if any(sample.token_ids[0] != 0 for sample in experiment.samples):
        raise NeuralBigramValidationError("every sample must start at the boundary")


def experiment_metrics(
    experiment: NeuralBigramExperiment,
) -> dict[str, int | float | str]:
    """Return compact, JSON-safe Day 20 metrics."""

    validate_experiment(experiment)
    max_gradient_error = max(
        probe.absolute_error for probe in experiment.gradient_probes
    )
    return {
        "examples": experiment.dataset.sample_count,
        "vocabulary_size": experiment.dataset.vocabulary.size,
        "one_hot_shape": (
            f"{experiment.dataset.sample_count}x{experiment.dataset.vocabulary.size}"
        ),
        "weight_shape": (
            f"{experiment.dataset.vocabulary.size}x{experiment.dataset.vocabulary.size}"
        ),
        "smoothing": experiment.smoothing,
        "count_loss": experiment.count_loss,
        "exact_neural_loss": experiment.exact_neural_loss,
        "equivalence_error": abs(experiment.count_loss - experiment.exact_neural_loss),
        "initial_loss": experiment.initial_loss,
        "trained_loss": experiment.trained_loss,
        "loss_reduction_percent": 100.0
        * (experiment.initial_loss - experiment.trained_loss)
        / experiment.initial_loss,
        "initial_perplexity": perplexity(experiment.initial_loss),
        "trained_perplexity": perplexity(experiment.trained_loss),
        "gradient_probe_count": len(experiment.gradient_probes),
        "max_gradient_error": max_gradient_error,
        "sample_count": len(experiment.samples),
        "terminated_samples": sum(sample.terminated for sample in experiment.samples),
        "seed": experiment.seed,
    }


def experiment_payload(experiment: NeuralBigramExperiment) -> dict[str, object]:
    """Convert the deterministic experiment to JSON-compatible data."""

    validate_experiment(experiment)
    return {
        "metrics": experiment_metrics(experiment),
        "vocabulary": list(experiment.dataset.vocabulary.tokens),
        "inputs": experiment.dataset.inputs.tolist(),
        "targets": experiment.dataset.targets.tolist(),
        "smoothed_probabilities": experiment.smoothed_probabilities.tolist(),
        "exact_weights": experiment.exact_weights.tolist(),
        "trained_weights": experiment.trained_weights.tolist(),
        "training_trace": [
            {
                "step": checkpoint.step,
                "loss": checkpoint.loss,
                "gradient_norm": checkpoint.gradient_norm,
            }
            for checkpoint in experiment.training_trace
        ],
        "gradient_probes": [
            {
                "row": probe.row,
                "column": probe.column,
                "analytic": probe.analytic,
                "numeric": probe.numeric,
                "absolute_error": probe.absolute_error,
            }
            for probe in experiment.gradient_probes
        ],
        "samples": [
            {
                "text": sample.text,
                "token_ids": list(sample.token_ids),
                "transition_probabilities": list(sample.transition_probabilities),
                "terminated": sample.terminated,
            }
            for sample in experiment.samples
        ],
    }


def render_dataflow_mermaid(experiment: NeuralBigramExperiment) -> str:
    """Render the one-hot neural-bigram tensor flow as Mermaid."""

    validate_experiment(experiment)
    examples = experiment.dataset.sample_count
    size = experiment.dataset.vocabulary.size
    return "\n".join(
        (
            "flowchart LR",
            f'    ids["input ids ({examples},)"] --> onehot["one-hot ({examples}, {size})"]',
            f'    weights["weights ({size}, {size})"] --> matmul["matrix multiply"]',
            "    onehot --> matmul",
            f'    matmul --> logits["logits ({examples}, {size})"]',
            f'    logits --> softmax["softmax probabilities ({examples}, {size})"]',
            f'    targets["target ids ({examples},)"] --> nll["mean negative log-likelihood"]',
            "    softmax --> nll",
        )
    )


def render_neural_bigram_markdown(experiment: NeuralBigramExperiment) -> str:
    """Render a Day 20 report without claiming learner completion."""

    validate_experiment(experiment)
    metrics = experiment_metrics(experiment)
    lines = [
        "# Day 20 neural character bigram model",
        "",
        "This deterministic report is repository evidence only. It does not prove ",
        "that the scheduled lectures were watched or that the learner completed the type-along.",
        "",
        "## Tensor path",
        "",
        f"- Examples: {metrics['examples']}",
        f"- One-hot matrix: `{metrics['one_hot_shape']}`",
        f"- Weight matrix: `{metrics['weight_shape']}`",
        "",
        "```mermaid",
        render_dataflow_mermaid(experiment),
        "```",
        "",
        "## Count-to-neural equivalence",
        "",
        f"- Additive smoothing: {metrics['smoothing']}",
        f"- Smoothed count-model NLL: {float(metrics['count_loss']):.12f}",
        f"- Log-count neural-model NLL: {float(metrics['exact_neural_loss']):.12f}",
        f"- Absolute loss difference: {float(metrics['equivalence_error']):.3e}",
        "",
        "## Optimization and gradient checks",
        "",
        f"- Initial NLL: {float(metrics['initial_loss']):.9f}",
        f"- Trained NLL: {float(metrics['trained_loss']):.9f}",
        f"- Loss reduction: {float(metrics['loss_reduction_percent']):.2f}%",
        f"- Initial perplexity: {float(metrics['initial_perplexity']):.6f}",
        f"- Trained perplexity: {float(metrics['trained_perplexity']):.6f}",
        f"- Gradient probes: {metrics['gradient_probe_count']}",
        f"- Maximum gradient error: {float(metrics['max_gradient_error']):.3e}",
        "",
        "| Step | Mean NLL | Gradient norm |",
        "|--:|--:|--:|",
    ]
    for checkpoint in experiment.training_trace:
        lines.append(
            f"| {checkpoint.step} | {checkpoint.loss:.9f} | "
            f"{checkpoint.gradient_norm:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic neural-model samples",
            "",
        ]
    )
    for index, sample in enumerate(experiment.samples, start=1):
        state = "terminated" if sample.terminated else "max length reached"
        lines.append(f"{index}. `{sample.text}` ({state})")
    lines.append("")
    return "\n".join(lines)
