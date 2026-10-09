"""Day 35: auditable causal averaging primitives for self-attention."""

from __future__ import annotations

import torch
from torch import Tensor


class CausalAverageError(ValueError):
    """Raised when a causal-average input or evidence contract is invalid."""


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
