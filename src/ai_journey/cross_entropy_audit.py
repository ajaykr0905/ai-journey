"""Independent CPU gradient evidence for the manual categorical loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from .cross_entropy import manual_cross_entropy


@dataclass(frozen=True)
class GradientAudit:
    examples: int
    classes: int
    manual_loss: float
    reference_loss: float
    loss_error: float
    autograd_error: float
    finite_difference_error: float
    row_sum_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def audit_cross_entropy(
    logits: object,
    targets: object,
    *,
    epsilon: float = 1e-5,
    tolerance: float = 1e-8,
) -> GradientAudit:
    """Compare every logit derivative with CPU autograd and finite differences.

    This is a small-matrix diagnostic, not a training loss: finite differences
    require two extra forward passes per scalar logit.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    trace = manual_cross_entropy(logits, targets)
    values = np.array(logits, dtype=np.float64, copy=True)
    labels = np.array(targets, dtype=np.int64, copy=True)
    reference_logits = torch.tensor(
        values, dtype=torch.float64, device="cpu", requires_grad=True
    )
    reference_loss = F.cross_entropy(reference_logits, torch.from_numpy(labels))
    reference_loss.backward()
    numeric = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        if plus[index] == values[index] or minus[index] == values[index]:
            raise ValueError("epsilon cannot perturb logits at this magnitude")
        numeric[index] = (
            manual_cross_entropy(plus, labels).loss
            - manual_cross_entropy(minus, labels).loss
        ) / (2 * epsilon)
    loss_error = abs(trace.loss - float(reference_loss.detach()))
    autograd_error = float(
        np.max(np.abs(trace.dlogits - reference_logits.grad.numpy()))
    )
    finite_difference_error = float(np.max(np.abs(trace.dlogits - numeric)))
    row_sum_error = float(np.max(np.abs(trace.dlogits.sum(axis=1))))
    errors = (loss_error, autograd_error, finite_difference_error, row_sum_error)
    return GradientAudit(
        len(labels),
        values.shape[1],
        trace.loss,
        float(reference_loss.detach()),
        loss_error,
        autograd_error,
        finite_difference_error,
        row_sum_error,
        tolerance,
        all(math.isfinite(error) and error <= tolerance for error in errors),
    )
