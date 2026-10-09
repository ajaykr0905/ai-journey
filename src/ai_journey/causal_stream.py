"""Streaming causal averages for bounded-memory inference."""

from __future__ import annotations

import torch
from torch import Tensor

from .causal_average import CausalAverageError, validate_values


class CausalAverageStream:
    """Maintain prefix count and sum while processing consecutive chunks."""

    def __init__(
        self,
        feature_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(feature_size, bool) or not isinstance(feature_size, int):
            raise TypeError("feature_size must be an integer")
        if feature_size <= 0:
            raise CausalAverageError("feature_size must be positive")
        if not dtype.is_floating_point:
            raise CausalAverageError("stream dtype must be floating-point")
        self._sum = torch.zeros(feature_size, dtype=dtype, device=device)
        self._count = 0

    @property
    def count(self) -> int:
        """Number of valid positions consumed so far."""

        return self._count

    def update(self, chunk: Tensor) -> Tensor:
        """Return causal averages for one non-empty consecutive chunk."""

        validate_values(chunk)
        if chunk.ndim != 2:
            raise CausalAverageError("stream chunks must have shape (time, channels)")
        if chunk.shape[1] != self._sum.numel():
            raise CausalAverageError("chunk feature size does not match stream state")
        if chunk.dtype != self._sum.dtype or chunk.device != self._sum.device:
            raise CausalAverageError("chunk dtype and device must match stream state")
        cumulative = chunk.cumsum(dim=0) + self._sum
        counts = torch.arange(
            self._count + 1,
            self._count + len(chunk) + 1,
            dtype=chunk.dtype,
            device=chunk.device,
        ).unsqueeze(-1)
        output = cumulative / counts
        self._sum = cumulative[-1].detach().clone()
        self._count += len(chunk)
        return output
