"""Day 14: deterministic transformer-MLP memory and superposition experiments.

For an input row ``x``, a bias-free output can be expanded as::

    MLP(x) = sum_i relu(x @ key_i + bias_i) * value_i

The expansion is exact algebra.  Calling the columns of the first matrix
"keys" and the rows of the second matrix "values" is an interpretability lens,
not a claim that every neuron stores one clean fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MLPWeights:
    """Parameters for a two-matrix ReLU MLP."""

    keys: FloatArray
    key_bias: FloatArray
    values: FloatArray
    output_bias: FloatArray


@dataclass(frozen=True)
class MLPTrace:
    """Inspectable intermediates and per-neuron output contributions."""

    inputs: FloatArray
    preactivations: FloatArray
    activations: FloatArray
    contributions: FloatArray
    outputs: FloatArray


@dataclass(frozen=True)
class MemoryExperiment:
    """Results from the deterministic Day 14 teaching experiment."""

    weights: MLPWeights
    queries: FloatArray
    trace: MLPTrace
    direct_outputs: FloatArray
    transformed_queries: FloatArray
    transformed_weights: MLPWeights
    transformed_trace: MLPTrace
    feature_directions: FloatArray
    sparse_features: FloatArray
    colliding_features: FloatArray
    sparse_reconstruction: FloatArray
    collision_reconstruction: FloatArray


def _finite_array(name: str, value: object, *, ndim: int) -> FloatArray:
    """Return a finite float array with an exact rank."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def relu(values: object) -> FloatArray:
    """Apply the rectified linear unit elementwise."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be non-empty and finite")
    return np.maximum(array, 0.0)


def validate_mlp_weights(weights: MLPWeights) -> tuple[int, int, int]:
    """Validate parameter shapes and return model, hidden, and output widths."""

    keys = _finite_array("keys", weights.keys, ndim=2)
    key_bias = _finite_array("key_bias", weights.key_bias, ndim=1)
    values = _finite_array("values", weights.values, ndim=2)
    output_bias = _finite_array("output_bias", weights.output_bias, ndim=1)
    model_dim, hidden_dim = keys.shape
    if key_bias.shape != (hidden_dim,):
        raise ValueError("key_bias must have shape (hidden_dim,)")
    if values.shape[0] != hidden_dim:
        raise ValueError("values must have shape (hidden_dim, output_dim)")
    output_dim = values.shape[1]
    if output_bias.shape != (output_dim,):
        raise ValueError("output_bias must have shape (output_dim,)")
    return model_dim, hidden_dim, output_dim


def mlp_forward(inputs: object, weights: MLPWeights) -> MLPTrace:
    """Run a ReLU MLP and expose each hidden neuron's output contribution."""

    x = _finite_array("inputs", inputs, ndim=2)
    model_dim, _, _ = validate_mlp_weights(weights)
    if x.shape[1] != model_dim:
        raise ValueError("inputs must have shape (batch, model_dim)")
    preactivations = x @ weights.keys + weights.key_bias
    activations = relu(preactivations)
    contributions = activations[:, :, None] * weights.values[None, :, :]
    outputs = contributions.sum(axis=1) + weights.output_bias
    return MLPTrace(x, preactivations, activations, contributions, outputs)


def direct_mlp_outputs(inputs: object, weights: MLPWeights) -> FloatArray:
    """Compute the conventional two-matrix expression independently."""

    x = _finite_array("inputs", inputs, ndim=2)
    model_dim, _, _ = validate_mlp_weights(weights)
    if x.shape[1] != model_dim:
        raise ValueError("inputs must have shape (batch, model_dim)")
    return (
        relu(x @ weights.keys + weights.key_bias) @ weights.values + weights.output_bias
    )


def reconstruct_outputs(trace: MLPTrace, output_bias: object) -> FloatArray:
    """Reconstruct outputs by summing stored per-neuron contributions."""

    contributions = _finite_array("contributions", trace.contributions, ndim=3)
    bias = _finite_array("output_bias", output_bias, ndim=1)
    if contributions.shape[2] != bias.shape[0]:
        raise ValueError("output_bias width must match contribution width")
    return contributions.sum(axis=1) + bias


def max_reconstruction_error(trace: MLPTrace, output_bias: object) -> float:
    """Measure the exact key/value expansion against the stored MLP output."""

    reconstructed = reconstruct_outputs(trace, output_bias)
    outputs = _finite_array("outputs", trace.outputs, ndim=2)
    if reconstructed.shape != outputs.shape:
        raise ValueError("trace outputs and contributions have incompatible shapes")
    return float(np.max(np.abs(outputs - reconstructed)))


def contribution_norms(trace: MLPTrace) -> FloatArray:
    """Return the L2 magnitude contributed by each neuron for each input."""

    contributions = _finite_array("contributions", trace.contributions, ndim=3)
    return np.linalg.norm(contributions, axis=-1)


def top_contributing_neurons(trace: MLPTrace, count: int = 1) -> NDArray[np.int64]:
    """Rank hidden neurons by contribution magnitude for each input row."""

    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    norms = contribution_norms(trace)
    if count > norms.shape[1]:
        raise ValueError("count cannot exceed the hidden width")
    return np.argsort(-norms, axis=1, kind="stable")[:, :count]


def cosine_similarity(left: object, right: object) -> float:
    """Return cosine similarity for two non-zero vectors."""

    a = _finite_array("left", left, ndim=1)
    b = _finite_array("right", right, ndim=1)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / denominator)


def change_basis(
    inputs: object, weights: MLPWeights, basis: object
) -> tuple[FloatArray, MLPWeights]:
    """Change input coordinates while preserving every key preactivation.

    Row vectors transform as ``x' = x B``.  Keys therefore transform as
    ``K' = B^-1 K``, making ``x' K' = x K``.
    """

    x = _finite_array("inputs", inputs, ndim=2)
    matrix = _finite_array("basis", basis, ndim=2)
    model_dim, _, _ = validate_mlp_weights(weights)
    if x.shape[1] != model_dim or matrix.shape != (model_dim, model_dim):
        raise ValueError("basis and inputs must match model_dim")
    if np.linalg.matrix_rank(matrix) != model_dim:
        raise ValueError("basis must be invertible")
    transformed_weights = MLPWeights(
        keys=np.linalg.solve(matrix, weights.keys),
        key_bias=weights.key_bias.copy(),
        values=weights.values.copy(),
        output_bias=weights.output_bias.copy(),
    )
    return x @ matrix, transformed_weights


def make_fact_memory() -> tuple[FloatArray, MLPWeights]:
    """Create four public-safe toy queries with four distinct output values."""

    queries = np.eye(4, dtype=np.float64)
    values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    weights = MLPWeights(
        keys=np.eye(4, dtype=np.float64),
        key_bias=np.full(4, -0.5, dtype=np.float64),
        values=values,
        output_bias=np.zeros(3, dtype=np.float64),
    )
    return queries, weights


def regular_feature_directions(
    feature_count: int, representation_dim: int = 2
) -> FloatArray:
    """Place feature directions evenly on a unit circle in two dimensions."""

    for name, value in (
        ("feature_count", feature_count),
        ("representation_dim", representation_dim),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if representation_dim != 2:
        raise ValueError("this visual toy experiment requires representation_dim=2")
    angles = 2.0 * pi * np.arange(feature_count, dtype=np.float64) / feature_count
    return np.vstack(
        ([cos(angle) for angle in angles], [sin(angle) for angle in angles])
    )


def feature_gram_matrix(directions: object) -> FloatArray:
    """Return pairwise feature-direction dot products."""

    matrix = _finite_array("directions", directions, ndim=2)
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("feature directions must be non-zero")
    normalized = matrix / norms
    return normalized.T @ normalized


def max_feature_coherence(directions: object) -> float:
    """Return the largest absolute dot product between distinct directions."""

    gram = feature_gram_matrix(directions)
    if gram.shape[0] < 2:
        return 0.0
    off_diagonal = gram - np.eye(gram.shape[0], dtype=np.float64)
    return float(np.max(np.abs(off_diagonal)))


def superposition_reconstruct(
    features: object, directions: object, *, bias: float = 0.25
) -> FloatArray:
    """Encode then decode non-negative features through a smaller representation."""

    x = _finite_array("features", features, ndim=2)
    matrix = _finite_array("directions", directions, ndim=2)
    if x.shape[1] != matrix.shape[1]:
        raise ValueError("feature width must match the number of directions")
    if not np.isfinite(bias) or bias < 0:
        raise ValueError("bias must be a finite, non-negative number")
    representation = x @ matrix.T
    return relu(representation @ matrix - bias)


def mean_squared_error(actual: object, expected: object) -> float:
    """Return mean squared error for arrays with matching shapes."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("arrays must be non-empty and have matching shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("arrays must contain only finite values")
    return float(np.mean((left - right) ** 2))


def run_memory_experiment() -> MemoryExperiment:
    """Run the deterministic MLP memory, basis, and superposition examples."""

    queries, weights = make_fact_memory()
    trace = mlp_forward(queries, weights)
    direct_outputs = direct_mlp_outputs(queries, weights)
    basis = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.3, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformed_queries, transformed_weights = change_basis(queries, weights, basis)
    transformed_trace = mlp_forward(transformed_queries, transformed_weights)

    directions = regular_feature_directions(5)
    sparse_features = np.eye(5, dtype=np.float64)
    colliding_features = np.array([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    sparse_reconstruction = superposition_reconstruct(sparse_features, directions)
    collision_reconstruction = superposition_reconstruct(colliding_features, directions)
    return MemoryExperiment(
        weights,
        queries,
        trace,
        direct_outputs,
        transformed_queries,
        transformed_weights,
        transformed_trace,
        directions,
        sparse_features,
        colliding_features,
        sparse_reconstruction,
        collision_reconstruction,
    )


def experiment_metrics(experiment: MemoryExperiment) -> dict[str, float | bool]:
    """Summarize exact identities and toy superposition interference."""

    direct_error = float(
        np.max(np.abs(experiment.trace.outputs - experiment.direct_outputs))
    )
    basis_error = float(
        np.max(np.abs(experiment.trace.outputs - experiment.transformed_trace.outputs))
    )
    sparse_error = mean_squared_error(
        experiment.sparse_reconstruction, experiment.sparse_features
    )
    collision_error = mean_squared_error(
        experiment.collision_reconstruction, experiment.colliding_features
    )
    return {
        "key_value_expansion_exact": max_reconstruction_error(
            experiment.trace, experiment.weights.output_bias
        )
        <= 1e-12,
        "direct_max_error": direct_error,
        "basis_invariance_max_error": basis_error,
        "feature_coherence": max_feature_coherence(experiment.feature_directions),
        "sparse_reconstruction_mse": sparse_error,
        "collision_reconstruction_mse": collision_error,
        "collision_increases_error": collision_error > sparse_error,
    }


def render_memory_markdown(
    experiment: MemoryExperiment, metrics: dict[str, float | bool]
) -> str:
    """Render an inspectable Markdown report for the Day 14 experiment."""

    top = top_contributing_neurons(experiment.trace).reshape(-1).tolist()
    return f"""# Day 14 MLP memory experiment

## Exact MLP decomposition

For each input row `x`:

```text
preactivation[i] = x @ key[i] + bias[i]
activation[i] = ReLU(preactivation[i])
output = sum_i activation[i] * value[i] + output_bias
```

- query shape: `{experiment.queries.shape}`
- key matrix shape: `{experiment.weights.keys.shape}`
- value matrix shape: `{experiment.weights.values.shape}`
- per-neuron contribution shape: `{experiment.trace.contributions.shape}`
- top contributing neuron per query: `{top}`
- direct-versus-expanded maximum error: `{float(metrics['direct_max_error']):.3e}`
- changed-basis output maximum error: `{float(metrics['basis_invariance_max_error']):.3e}`

The key/value phrasing is an algebraic and interpretability lens. This toy setup
deliberately gives each query one clean neuron. It does not show that real model
facts live in one neuron or that the weights were learned.

## Superposition toy model

Five feature directions share a two-dimensional representation. The decoder is
`ReLU(W.T @ W @ features - bias)`.

- maximum pairwise feature coherence: `{float(metrics['feature_coherence']):.6f}`
- one-feature-at-a-time reconstruction MSE: `{float(metrics['sparse_reconstruction_mse']):.6f}`
- two-feature collision reconstruction MSE: `{float(metrics['collision_reconstruction_mse']):.6f}`
- collision increases error: `{str(bool(metrics['collision_increases_error'])).lower()}`

The small experiment demonstrates a capacity/interference tradeoff. It is not
evidence about where any particular fact resides in a production language model.
"""
