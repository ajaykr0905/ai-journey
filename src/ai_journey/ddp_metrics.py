"""Token-weighted reductions used by replicated training and evaluation."""

import math

import torch

from .ddp_contracts import integer


def ddp_loss_scale(local_tokens: int, global_tokens: int, world_size: int) -> float:
    """Compensate for DDP's rank averaging when local target counts differ."""
    integer(local_tokens, "local_tokens")
    integer(global_tokens, "global_tokens", 1)
    integer(world_size, "world_size", 1)
    if local_tokens > global_tokens:
        raise ValueError("local_tokens exceeds global_tokens")
    return world_size * local_tokens / global_tokens


def weighted_mean(totals: list[tuple[float, int]]) -> float:
    if not totals:
        raise ValueError("at least one total is required")
    for loss_sum, tokens in totals:
        integer(tokens, "tokens")
        if isinstance(loss_sum, bool) or not math.isfinite(loss_sum) or loss_sum < 0:
            raise ValueError("loss sums must be finite and nonnegative")
    count = sum(tokens for _, tokens in totals)
    if count == 0:
        raise ValueError("at least one target token is required")
    return math.fsum(loss for loss, _ in totals) / count


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute per-target stable cross entropy without reducing padding weights."""
    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or targets.dtype != torch.long
    ):
        raise ValueError("logits and targets must have matching batch/time dimensions")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point values")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[-1]:
        raise ValueError("targets must be nonempty valid vocabulary ids")
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    if not bool(torch.isfinite(losses).all()):
        raise ValueError("token losses must be finite after cross entropy")
    return losses


def binary_metrics(labels, predictions) -> dict:
    """Expose sensitivity and specificity with undefined denominators explicit."""
    labels, predictions = tuple(labels), tuple(predictions)
    if (
        not labels
        or len(labels) != len(predictions)
        or any(
            type(value) is not int or value not in (0, 1)
            for value in (*labels, *predictions)
        )
    ):
        raise ValueError(
            "labels and predictions must be matching nonempty binary integers"
        )
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(
        label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    fn = sum(
        label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)
    )
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def calibration_metrics(probabilities, labels, *, bins: int = 10) -> dict:
    """Compute multiclass Brier score and empirical confidence-bin error."""
    integer(bins, "bins", 1)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or probabilities.shape[1] < 2
        or labels.dtype != torch.long
        or labels.shape != (probabilities.shape[0],)
    ):
        raise ValueError("probabilities and integer labels must have matching examples")
    if (
        not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not torch.allclose(
            probabilities.sum(1),
            torch.ones(probabilities.shape[0], dtype=torch.float64),
            atol=1e-10,
            rtol=0,
        )
    ):
        raise ValueError("each probability row must sum to one")
    if labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("label is outside the probability vocabulary")
    targets = torch.nn.functional.one_hot(labels, probabilities.shape[1]).double()
    confidence, prediction = probabilities.max(1)
    correct = (prediction == labels).double()
    assigned = (confidence * bins).long().clamp(max=bins - 1)
    records = []
    error = 0.0
    for index in range(bins):
        mask = assigned == index
        count = int(mask.sum())
        if count:
            mean_confidence = confidence[mask].mean().item()
            accuracy = correct[mask].mean().item()
            error += count / len(labels) * abs(mean_confidence - accuracy)
            records.append(
                {
                    "bin": index,
                    "count": count,
                    "confidence": mean_confidence,
                    "accuracy": accuracy,
                }
            )
    return {
        "brier_score": ((probabilities - targets) ** 2).sum(1).mean().item(),
        "expected_calibration_error": error,
        "bins": records,
        "examples": len(labels),
    }
