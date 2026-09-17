"""Day 13: deterministic, single-head scaled dot-product attention in NumPy.

The implementation favors explicit intermediate tensors over brevity.  It is a
small teaching reference for the equations ``Q = XWq``, ``K = XWk``,
``V = XWv``, and ``softmax(QK.T / sqrt(d))V``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class AttentionConfig:
    """Dimensions and controls for one deterministic attention experiment."""

    sequence_length: int = 4
    model_dim: int = 6
    head_dim: int = 3
    seed: int = 13
    causal: bool = True

    def __post_init__(self) -> None:
        for name in ("sequence_length", "model_dim", "head_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.causal, bool):
            raise ValueError("causal must be a boolean")


@dataclass(frozen=True)
class ProjectionWeights:
    """Query, key, and value projection matrices."""

    query: FloatArray
    key: FloatArray
    value: FloatArray


@dataclass(frozen=True)
class AttentionTrace:
    """Every intermediate tensor in a single-head attention calculation."""

    inputs: FloatArray
    queries: FloatArray
    keys: FloatArray
    values: FloatArray
    scale: float
    scores: FloatArray
    allowed: BoolArray
    weights: FloatArray
    context: FloatArray


def _float_matrix(name: str, value: object) -> FloatArray:
    """Normalize and validate a finite, non-empty floating-point matrix."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def stable_softmax(logits: object, axis: int = -1) -> FloatArray:
    """Compute softmax after subtracting the maximum for numerical stability."""

    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        raise ValueError("logits must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("logits must contain only finite values")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ValueError(f"invalid softmax axis: {axis}")
    normalized_axis = axis if axis >= 0 else values.ndim + axis
    if normalized_axis < 0 or normalized_axis >= values.ndim:
        raise ValueError(f"invalid softmax axis: {axis}")
    maxima = np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(values - maxima)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def causal_attention_mask(
    query_length: int, key_length: int | None = None
) -> BoolArray:
    """Return a causal mask, including prefix keys during incremental decoding.

    When ``key_length`` is longer than ``query_length``, the extra keys are
    treated as an already-seen prefix.  Every query can attend to that prefix
    and to its own position, but never to a later key.
    """

    key_length = query_length if key_length is None else key_length
    for name, value in (("query_length", query_length), ("key_length", key_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if key_length < query_length:
        raise ValueError("key_length must be at least query_length")
    prefix_length = key_length - query_length
    query_positions = np.arange(query_length)[:, None] + prefix_length
    key_positions = np.arange(key_length)[None, :]
    return key_positions <= query_positions


def masked_softmax(scores: object, allowed: object, axis: int = -1) -> FloatArray:
    """Normalize scores while assigning exactly zero probability to masked keys."""

    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.size == 0 or not np.isfinite(score_matrix).all():
        raise ValueError("scores must be non-empty and finite")
    mask = np.asarray(allowed)
    if mask.dtype != np.bool_:
        raise ValueError("allowed mask must contain booleans")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ValueError("mask or axis is incompatible with scores")
    normalized_axis = axis if axis >= 0 else score_matrix.ndim + axis
    if normalized_axis < 0 or normalized_axis >= score_matrix.ndim:
        raise ValueError("mask or axis is incompatible with scores")
    try:
        broadcast_mask = np.broadcast_to(mask, score_matrix.shape)
        has_allowed = np.any(broadcast_mask, axis=axis, keepdims=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("mask or axis is incompatible with scores") from exc
    if not np.all(has_allowed):
        raise ValueError("every softmax row must allow at least one key")

    masked_scores = np.where(broadcast_mask, score_matrix, -np.inf)
    maxima = np.max(masked_scores, axis=axis, keepdims=True)
    exponentials = np.where(broadcast_mask, np.exp(masked_scores - maxima), 0.0)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def project_qkv(
    inputs: object, weights: ProjectionWeights
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Project a ``(T, C)`` input into query, key, and value matrices."""

    x = _float_matrix("inputs", inputs)
    matrices = tuple(
        _float_matrix(name, matrix)
        for name, matrix in (
            ("query weights", weights.query),
            ("key weights", weights.key),
            ("value weights", weights.value),
        )
    )
    model_dim = x.shape[1]
    if any(matrix.shape[0] != model_dim for matrix in matrices):
        raise ValueError("each projection must have shape (model_dim, head_dim)")
    head_dims = {matrix.shape[1] for matrix in matrices}
    if len(head_dims) != 1:
        raise ValueError("query, key, and value projections must share head_dim")
    return tuple(x @ matrix for matrix in matrices)  # type: ignore[return-value]


def scaled_dot_product_scores(
    queries: object, keys: object
) -> tuple[FloatArray, float]:
    """Compute ``QK.T / sqrt(head_dim)`` and return the applied scale."""

    q = _float_matrix("queries", queries)
    k = _float_matrix("keys", keys)
    if q.shape[1] != k.shape[1]:
        raise ValueError("queries and keys must share head_dim")
    scale = 1.0 / sqrt(q.shape[1])
    return (q @ k.T) * scale, scale


def weighted_value_sum(weights: object, values: object) -> FloatArray:
    """Combine value vectors using one normalized attention row per query."""

    probabilities = _float_matrix("attention weights", weights)
    value_matrix = _float_matrix("values", values)
    if probabilities.shape[1] != value_matrix.shape[0]:
        raise ValueError("attention key dimension must match the number of values")
    if np.any(probabilities < 0):
        raise ValueError("attention weights must be non-negative")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("each attention row must sum to one")
    return probabilities @ value_matrix


def single_head_attention(
    inputs: object,
    projection_weights: ProjectionWeights,
    *,
    causal: bool = True,
) -> AttentionTrace:
    """Run one explicit, vectorized scaled dot-product attention head."""

    if not isinstance(causal, bool):
        raise ValueError("causal must be a boolean")
    x = _float_matrix("inputs", inputs)
    queries, keys, values = project_qkv(x, projection_weights)
    scores, scale = scaled_dot_product_scores(queries, keys)
    allowed = (
        causal_attention_mask(queries.shape[0], keys.shape[0])
        if causal
        else np.ones(scores.shape, dtype=np.bool_)
    )
    weights = masked_softmax(scores, allowed)
    context = weighted_value_sum(weights, values)
    return AttentionTrace(
        inputs=x,
        queries=queries,
        keys=keys,
        values=values,
        scale=scale,
        scores=scores,
        allowed=allowed,
        weights=weights,
        context=context,
    )


def make_random_attention_problem(
    config: AttentionConfig,
) -> tuple[FloatArray, ProjectionWeights]:
    """Create deterministic random inputs and variance-scaled projections."""

    generator = np.random.default_rng(config.seed)
    inputs = generator.standard_normal((config.sequence_length, config.model_dim))
    weight_scale = 1.0 / sqrt(config.model_dim)
    matrices = [
        generator.standard_normal((config.model_dim, config.head_dim)) * weight_scale
        for _ in range(3)
    ]
    return inputs, ProjectionWeights(*matrices)


def run_attention_experiment(config: AttentionConfig | None = None) -> AttentionTrace:
    """Build and execute the default deterministic Day 13 experiment."""

    cfg = config or AttentionConfig()
    inputs, projections = make_random_attention_problem(cfg)
    return single_head_attention(inputs, projections, causal=cfg.causal)


def attention_invariants(trace: AttentionTrace) -> dict[str, float | bool]:
    """Measure the numerical and causal properties expected from a valid trace."""

    matrices_are_2d = all(
        matrix.ndim == 2
        for matrix in (
            trace.inputs,
            trace.queries,
            trace.keys,
            trace.values,
            trace.scores,
            trace.allowed,
            trace.weights,
            trace.context,
        )
    )
    expected_scores = (
        (trace.queries.shape[0], trace.keys.shape[0]) if matrices_are_2d else ()
    )
    expected_context = (
        (trace.queries.shape[0], trace.values.shape[1]) if matrices_are_2d else ()
    )
    shape_ok = (
        matrices_are_2d
        and trace.scores.shape == expected_scores
        and trace.allowed.shape == expected_scores
        and trace.weights.shape == expected_scores
        and trace.context.shape == expected_context
    )
    all_finite = all(
        np.isfinite(matrix).all()
        for matrix in (
            trace.inputs,
            trace.queries,
            trace.keys,
            trace.values,
            trace.scores,
            trace.weights,
            trace.context,
        )
    )
    if trace.weights.ndim == 2 and trace.weights.size:
        row_sum_error = float(np.max(np.abs(trace.weights.sum(axis=-1) - 1.0)))
        minimum_weight = float(np.min(trace.weights))
    else:
        row_sum_error = float("inf")
        minimum_weight = float("-inf")
    if trace.weights.shape == trace.allowed.shape:
        masked_weights = trace.weights[~trace.allowed]
        maximum_masked_weight = (
            float(np.max(masked_weights)) if masked_weights.size else 0.0
        )
    else:
        maximum_masked_weight = float("inf")
    return {
        "shape_ok": bool(shape_ok),
        "all_finite": bool(all_finite),
        "max_row_sum_error": row_sum_error,
        "max_masked_weight": maximum_masked_weight,
        "min_weight": minimum_weight,
    }


def validate_attention_trace(
    trace: AttentionTrace, atol: float = 1e-12
) -> dict[str, float | bool]:
    """Reject a trace that violates shape, probability, or masking invariants."""

    if atol < 0 or not np.isfinite(atol):
        raise ValueError("atol must be a finite, non-negative number")
    metrics = attention_invariants(trace)
    if not metrics["shape_ok"]:
        raise ValueError("attention trace contains an invalid tensor shape")
    if not metrics["all_finite"]:
        raise ValueError("attention trace contains a non-finite value")
    if float(metrics["min_weight"]) < -atol:
        raise ValueError("attention weights must be non-negative")
    if float(metrics["max_row_sum_error"]) > atol:
        raise ValueError("attention rows must sum to one")
    if float(metrics["max_masked_weight"]) > atol:
        raise ValueError("masked positions must have zero attention weight")
    return metrics


def reference_attention_loop(
    inputs: object,
    projection_weights: ProjectionWeights,
    *,
    causal: bool = True,
) -> FloatArray:
    """Compute the same result with loops as an independent teaching oracle."""

    x = _float_matrix("inputs", inputs)
    queries, keys, values = project_qkv(x, projection_weights)
    allowed = (
        causal_attention_mask(queries.shape[0], keys.shape[0])
        if causal
        else np.ones((queries.shape[0], keys.shape[0]), dtype=np.bool_)
    )
    output = np.zeros((queries.shape[0], values.shape[1]), dtype=np.float64)
    scale = 1.0 / sqrt(queries.shape[1])
    for query_index in range(queries.shape[0]):
        key_indexes = np.flatnonzero(allowed[query_index])
        row_scores = np.array(
            [
                float(np.dot(queries[query_index], keys[key_index])) * scale
                for key_index in key_indexes
            ],
            dtype=np.float64,
        )
        row_weights = stable_softmax(row_scores)
        for weight, key_index in zip(row_weights, key_indexes):
            output[query_index] += weight * values[key_index]
    return output


def max_reference_error(
    trace: AttentionTrace,
    inputs: object,
    projection_weights: ProjectionWeights,
    *,
    causal: bool,
) -> float:
    """Return the largest difference from the independent loop implementation."""

    reference = reference_attention_loop(inputs, projection_weights, causal=causal)
    return float(np.max(np.abs(trace.context - reference)))


def format_matrix(matrix: object, precision: int = 6) -> str:
    """Render a small matrix deterministically for lesson artifacts."""

    values = _float_matrix("matrix", matrix)
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 0:
        raise ValueError("precision must be a non-negative integer")
    return np.array2string(
        values,
        precision=precision,
        suppress_small=False,
        floatmode="fixed",
    )


def render_attention_markdown(
    config: AttentionConfig,
    trace: AttentionTrace,
    metrics: dict[str, float | bool],
    reference_error: float,
) -> str:
    """Render the deterministic experiment as an inspectable Markdown report."""

    return f"""# Day 13 single-head attention experiment

## Configuration

- sequence length: `{config.sequence_length}`
- model width: `{config.model_dim}`
- head width: `{config.head_dim}`
- seed: `{config.seed}`
- causal mask: `{str(config.causal).lower()}`
- score scale: `1 / sqrt({config.head_dim}) = {trace.scale:.12f}`

## Equations and shapes

1. `Q = X Wq`, `K = X Wk`, `V = X Wv`: `(T, C) @ (C, D) -> (T, D)`
2. `S = Q K.T / sqrt(D)`: `(T, D) @ (D, T) -> (T, T)`
3. `A = masked_softmax(S)`: `(T, T) -> (T, T)`
4. `Y = A V`: `(T, T) @ (T, D) -> (T, D)`

## Attention weights

```text
{format_matrix(trace.weights)}
```

## Context vectors

```text
{format_matrix(trace.context)}
```

## Validation

- shapes valid: `{str(bool(metrics['shape_ok'])).lower()}`
- all values finite: `{str(bool(metrics['all_finite'])).lower()}`
- maximum row-sum error: `{float(metrics['max_row_sum_error']):.3e}`
- maximum masked weight: `{float(metrics['max_masked_weight']):.3e}`
- vectorized-versus-loop maximum error: `{reference_error:.3e}`
"""
