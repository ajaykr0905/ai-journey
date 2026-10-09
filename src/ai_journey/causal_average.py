"""Day 35: auditable causal averaging primitives for self-attention."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

import torch
from torch import Tensor


class CausalAverageError(ValueError):
    """Raised when a causal-average input or evidence contract is invalid."""


@dataclass(frozen=True)
class CausalAverageAudit:
    """Maximum forward differences against the explicit loop oracle."""

    matmul_error: float
    softmax_error: float
    cumsum_error: float


@dataclass(frozen=True)
class CausalWeightAudit:
    """Structural invariants for one square causal weight matrix."""

    row_sum_error: float
    maximum_future_weight: float
    minimum_weight: float


@dataclass(frozen=True)
class CausalGradientAudit:
    """Maximum input-gradient differences against the explicit loop oracle."""

    matmul_error: float
    softmax_error: float
    cumsum_error: float


def validate_values(values: Tensor) -> None:
    """Validate a feature sequence shaped ``(time, channels)`` or batched equivalent."""

    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim not in (2, 3):
        raise CausalAverageError(
            "values must have shape (time, channels) or (batch, time, channels)"
        )
    if values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise CausalAverageError("time and channel dimensions must be non-empty")
    if not values.is_floating_point():
        raise CausalAverageError("values must use a floating-point dtype")
    if not torch.isfinite(values).all():
        raise CausalAverageError("values must be finite")


def causal_mask(length: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return a lower-triangular boolean mask for a positive sequence length."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length <= 0:
        raise CausalAverageError("length must be positive")
    return torch.ones((length, length), dtype=torch.bool, device=device).tril()


def causal_average_loop(values: Tensor) -> Tensor:
    """Average every prefix with an explicit loop that serves as the oracle."""

    validate_values(values)
    time_dimension = values.shape[-2]
    prefixes = [
        values[..., : index + 1, :].mean(dim=-2) for index in range(time_dimension)
    ]
    return torch.stack(prefixes, dim=-2)


def triangular_average_weights(
    length: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the normalized lower-triangular matrix of prefix weights."""

    mask = causal_mask(length, device=device)
    if not dtype.is_floating_point:
        raise CausalAverageError("weight dtype must be floating-point")
    weights = mask.to(dtype=dtype)
    return weights / weights.sum(dim=-1, keepdim=True)


def causal_average_matmul(values: Tensor) -> Tensor:
    """Apply normalized lower-triangular weights with one matrix multiply."""

    validate_values(values)
    weights = triangular_average_weights(
        values.shape[-2], dtype=values.dtype, device=values.device
    )
    return torch.matmul(weights, values)


def masked_softmax_weights(
    length: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build uniform causal weights as masked softmax over zero logits."""

    mask = causal_mask(length, device=device)
    if not dtype.is_floating_point:
        raise CausalAverageError("weight dtype must be floating-point")
    logits = torch.zeros((length, length), dtype=dtype, device=device)
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def causal_average_softmax(values: Tensor) -> Tensor:
    """Apply masked-softmax causal weights to a feature sequence."""

    validate_values(values)
    weights = masked_softmax_weights(
        values.shape[-2], dtype=values.dtype, device=values.device
    )
    return torch.matmul(weights, values)


def causal_average_cumsum(values: Tensor) -> Tensor:
    """Compute prefix means in linear memory with a cumulative sum."""

    validate_values(values)
    length = values.shape[-2]
    counts = torch.arange(
        1, length + 1, dtype=values.dtype, device=values.device
    ).unsqueeze(-1)
    return values.cumsum(dim=-2) / counts


def audit_equivalence(values: Tensor) -> CausalAverageAudit:
    """Measure every vectorized method against the explicit prefix loop."""

    reference = causal_average_loop(values)

    def maximum_error(candidate: Tensor) -> float:
        return float((candidate - reference).abs().max().item())

    return CausalAverageAudit(
        matmul_error=maximum_error(causal_average_matmul(values)),
        softmax_error=maximum_error(causal_average_softmax(values)),
        cumsum_error=maximum_error(causal_average_cumsum(values)),
    )


def require_equivalence(audit: CausalAverageAudit, *, tolerance: float) -> None:
    """Reject a forward audit when any implementation exceeds its declared tolerance."""

    if not isinstance(audit, CausalAverageAudit):
        raise TypeError("audit must be CausalAverageAudit")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not isfinite(tolerance)
        or tolerance < 0
    ):
        raise CausalAverageError("tolerance must be finite and non-negative")
    errors = {
        "matmul": audit.matmul_error,
        "softmax": audit.softmax_error,
        "cumsum": audit.cumsum_error,
    }
    for name, error in errors.items():
        if not isfinite(error) or error > tolerance:
            raise CausalAverageError(
                f"{name} forward error {error:.3e} exceeds tolerance {tolerance:.3e}"
            )


def audit_weights(weights: Tensor) -> CausalWeightAudit:
    """Measure normalization, causality, and non-negativity of square weights."""

    if not isinstance(weights, Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if (
        weights.ndim != 2
        or weights.shape[0] != weights.shape[1]
        or weights.shape[0] == 0
    ):
        raise CausalAverageError("weights must be a non-empty square matrix")
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise CausalAverageError("weights must be finite floating-point values")
    mask = causal_mask(weights.shape[0], device=weights.device)
    future = weights[~mask]
    maximum_future_weight = float(future.abs().max().item()) if future.numel() else 0.0
    return CausalWeightAudit(
        row_sum_error=float((weights.sum(dim=-1) - 1).abs().max().item()),
        maximum_future_weight=maximum_future_weight,
        minimum_weight=float(weights.min().item()),
    )


def require_weight_safety(audit: CausalWeightAudit, *, tolerance: float) -> None:
    """Reject weights that are unnormalized, non-causal, or materially negative."""

    if not isinstance(audit, CausalWeightAudit):
        raise TypeError("audit must be CausalWeightAudit")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not isfinite(tolerance)
        or tolerance < 0
    ):
        raise CausalAverageError("tolerance must be finite and non-negative")
    if audit.row_sum_error > tolerance:
        raise CausalAverageError("causal weight rows do not sum to one")
    if audit.maximum_future_weight > tolerance:
        raise CausalAverageError("causal weights leak future positions")
    if audit.minimum_weight < -tolerance:
        raise CausalAverageError("causal weights contain negative probability mass")


def future_influence_error(
    values: Tensor,
    method: Callable[[Tensor], Tensor],
    *,
    perturbation: float = 1000.0,
) -> float:
    """Measure whether changing a suffix alters outputs that precede that suffix."""

    validate_values(values)
    if not callable(method):
        raise TypeError("method must be callable")
    if (
        isinstance(perturbation, bool)
        or not isinstance(perturbation, (int, float))
        or not isfinite(perturbation)
        or perturbation == 0
    ):
        raise CausalAverageError("perturbation must be finite and non-zero")
    reference = method(values)
    if not isinstance(reference, Tensor) or reference.shape != values.shape:
        raise CausalAverageError("method must preserve the input tensor shape")
    maximum = 0.0
    for cutoff in range(1, values.shape[-2]):
        changed = values.detach().clone()
        changed[..., cutoff:, :] += perturbation
        candidate = method(changed)
        error = float(
            (candidate[..., :cutoff, :] - reference[..., :cutoff, :]).abs().max()
        )
        maximum = max(maximum, error)
    return maximum


def audit_gradient_equivalence(values: Tensor) -> CausalGradientAudit:
    """Compare input gradients under a deterministic non-uniform upstream tensor."""

    validate_values(values)
    upstream = torch.linspace(
        -0.75,
        1.25,
        values.numel(),
        dtype=values.dtype,
        device=values.device,
    ).reshape(values.shape)

    def gradient(method: Callable[[Tensor], Tensor]) -> Tensor:
        candidate = values.detach().clone().requires_grad_(True)
        (method(candidate) * upstream).sum().backward()
        if candidate.grad is None:
            raise CausalAverageError("method did not produce an input gradient")
        return candidate.grad.detach()

    reference = gradient(causal_average_loop)

    def maximum_error(method: Callable[[Tensor], Tensor]) -> float:
        return float((gradient(method) - reference).abs().max().item())

    return CausalGradientAudit(
        matmul_error=maximum_error(causal_average_matmul),
        softmax_error=maximum_error(causal_average_softmax),
        cumsum_error=maximum_error(causal_average_cumsum),
    )


def validate_lengths(values: Tensor, lengths: Tensor) -> None:
    """Validate per-example lengths for a padded rank-three batch."""

    validate_values(values)
    if values.ndim != 3:
        raise CausalAverageError(
            "padded values must have shape (batch, time, channels)"
        )
    if not isinstance(lengths, Tensor):
        raise TypeError("lengths must be a torch.Tensor")
    if lengths.device != values.device:
        raise CausalAverageError("lengths must be on the same device as values")
    if lengths.ndim != 1 or lengths.shape[0] != values.shape[0]:
        raise CausalAverageError("lengths must contain one entry per batch item")
    if lengths.dtype != torch.long:
        raise CausalAverageError("lengths must use torch.long")
    if torch.any(lengths <= 0) or torch.any(lengths > values.shape[1]):
        raise CausalAverageError(
            "lengths must be between one and the padded time dimension"
        )


def causal_average_padded(values: Tensor, lengths: Tensor) -> Tensor:
    """Average valid prefixes and force padded output positions to exact zero."""

    validate_lengths(values, lengths)
    output = causal_average_cumsum(values)
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    return output.masked_fill(~valid.unsqueeze(-1), 0.0)


def padding_influence_error(
    values: Tensor, lengths: Tensor, *, perturbation: float = 1000.0
) -> float:
    """Measure whether padded values can change any valid causal output."""

    validate_lengths(values, lengths)
    if (
        isinstance(perturbation, bool)
        or not isinstance(perturbation, (int, float))
        or not isfinite(perturbation)
        or perturbation == 0
    ):
        raise CausalAverageError("perturbation must be finite and non-zero")
    reference = causal_average_padded(values, lengths)
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    changed = values.detach().clone()
    changed.masked_fill_(~valid.unsqueeze(-1), perturbation)
    candidate = causal_average_padded(changed, lengths)
    return float((candidate[valid] - reference[valid]).abs().max().item())
