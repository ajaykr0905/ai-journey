"""Stable categorical cross-entropy with an inspectable manual backward pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CrossEntropyTrace:
    """Forward intermediates and gradients for a mean loss over examples."""

    loss: float
    log_probabilities: Array
    probabilities: Array
    per_example_loss: Array
    dlog_probabilities: Array
    dlog_normalizer: Array
    dshifted_logits: Array
    dlogits: Array


def _validate(logits: object, targets: object) -> tuple[Array, NDArray[np.int64]]:
    raw = np.asarray(logits)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2:
        raise ValueError("logits must be a real numeric [examples, classes] matrix")
    if raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError("logits require at least one example and two classes")
    values = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    labels = np.asarray(targets)
    if labels.dtype.kind not in "iu" or labels.shape != (values.shape[0],):
        raise ValueError("targets must be an integer vector matching the examples")
    if np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("targets must index existing classes")
    return values, labels.astype(np.int64, copy=True)


def manual_cross_entropy(logits: object, targets: object) -> CrossEntropyTrace:
    """Evaluate stable mean NLL and backpropagate through log-sum-exp.

    Subtracting each row's maximum preserves the loss. Its derivative cancels
    because each row's shifted-logit gradient sums to zero; no argmax derivative
    is needed, including at ties. This implementation never divides by a target
    probability, which may underflow to zero for a confidently wrong prediction.
    """
    values, labels = _validate(logits, targets)
    with np.errstate(over="raise", invalid="raise"):
        try:
            shifted = values - values.max(axis=1, keepdims=True)
            counts = np.exp(shifted)
            normalizer = counts.sum(axis=1, keepdims=True)
            log_probabilities = shifted - np.log(normalizer)
        except FloatingPointError as exc:
            raise ValueError("logit range exceeds float64 arithmetic") from exc
    probabilities = counts / normalizer
    row = np.arange(len(labels))
    per_example_loss = -log_probabilities[row, labels]
    # Sum after division to avoid overflowing an otherwise representable mean.
    loss = float(np.sum(per_example_loss / len(labels)))
    if not np.isfinite(loss):
        raise ValueError("mean loss exceeds float64 arithmetic")

    dlog_probabilities = np.zeros_like(values)
    dlog_probabilities[row, labels] = -1.0 / len(labels)
    dlog_normalizer = -dlog_probabilities.sum(axis=1, keepdims=True)
    dnormalizer = dlog_normalizer / normalizer
    dcounts = np.broadcast_to(dnormalizer, counts.shape)
    dshifted_logits = dlog_probabilities + dcounts * counts
    return CrossEntropyTrace(
        loss=loss,
        log_probabilities=log_probabilities,
        probabilities=probabilities,
        per_example_loss=per_example_loss,
        dlog_probabilities=dlog_probabilities,
        dlog_normalizer=dlog_normalizer,
        dshifted_logits=dshifted_logits,
        dlogits=dshifted_logits.copy(),
    )
