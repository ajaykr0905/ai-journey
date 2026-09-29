"""Deterministic activation and gradient distribution diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor


class DiagnosticError(ValueError):
    """Raised when a diagnostic request or tensor is invalid."""


@dataclass(frozen=True)
class HistogramSpec:
    """Fixed histogram boundaries for comparable tensor snapshots."""

    lower: float = -5.0
    upper: float = 5.0
    bins: int = 40
    near_zero: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "near_zero"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise DiagnosticError(f"{name} must be finite")
        if self.lower >= self.upper:
            raise DiagnosticError("lower must be less than upper")
        if isinstance(self.bins, bool) or not isinstance(self.bins, int):
            raise TypeError("bins must be an integer")
        if self.bins <= 0:
            raise DiagnosticError("bins must be positive")
        if self.near_zero < 0:
            raise DiagnosticError("near_zero must be non-negative")


@dataclass(frozen=True)
class TensorDistribution:
    """Serializable scalar and histogram evidence for one tensor."""

    count: int
    finite_count: int
    nonfinite_count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    rms: float
    zero_fraction: float
    near_zero_fraction: float
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    underflow_count: int
    overflow_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_tensor(
    tensor: Tensor, spec: HistogramSpec | None = None
) -> TensorDistribution:
    """Summarize a tensor without retaining its raw values."""

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.numel() == 0:
        raise DiagnosticError("tensor must not be empty")
    spec = spec or HistogramSpec()
    values = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    if finite.numel() == 0:
        raise DiagnosticError("tensor has no finite values")
    below = int((finite < spec.lower).sum())
    above = int((finite > spec.upper).sum())
    counts = torch.histc(finite, bins=spec.bins, min=spec.lower, max=spec.upper)
    edges = torch.linspace(
        spec.lower, spec.upper, steps=spec.bins + 1, dtype=torch.float64
    )
    finite_count = int(finite.numel())
    return TensorDistribution(
        count=int(values.numel()),
        finite_count=finite_count,
        nonfinite_count=int((~finite_mask).sum()),
        mean=float(finite.mean()),
        standard_deviation=float(finite.std(correction=0)),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        rms=float(finite.square().mean().sqrt()),
        zero_fraction=float((finite == 0).sum()) / finite_count,
        near_zero_fraction=float((finite.abs() <= spec.near_zero).sum()) / finite_count,
        histogram_edges=tuple(float(value) for value in edges),
        histogram_counts=tuple(int(value) for value in counts),
        underflow_count=below,
        overflow_count=above,
    )
