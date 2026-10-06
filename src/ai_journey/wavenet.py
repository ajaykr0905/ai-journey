"""Hierarchical character-language-model primitives for curriculum Day 32."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import reduce
from hashlib import sha256
from operator import mul


class WaveNetError(ValueError):
    """Raised when hierarchical model data, configuration, or state is invalid."""


@dataclass(frozen=True)
class WaveNetConfig:
    """Validated architecture for a tree-structured character language model."""

    vocab_size: int
    context_size: int = 8
    embedding_dim: int = 16
    hidden_dim: int = 64
    group_factors: tuple[int, ...] = (2, 2, 2)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_size", "embedding_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            minimum = 2 if name == "vocab_size" else 1
            if value < minimum:
                raise WaveNetError(f"{name} must be at least {minimum}")

        if not isinstance(self.group_factors, tuple) or not self.group_factors:
            raise TypeError("group_factors must be a non-empty tuple of integers")
        if any(
            isinstance(factor, bool) or not isinstance(factor, int)
            for factor in self.group_factors
        ):
            raise TypeError("group_factors must contain only integers")
        if any(factor < 2 for factor in self.group_factors):
            raise WaveNetError("group_factors must be at least 2")
        if self.receptive_field != self.context_size:
            raise WaveNetError(
                "context_size must equal the product of group_factors"
            )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise WaveNetError("dropout must be finite and in [0, 1)")

    @property
    def receptive_field(self) -> int:
        """Return the number of input tokens consumed by the hierarchy."""

        return reduce(mul, self.group_factors, 1)

    @property
    def stage_lengths(self) -> tuple[int, ...]:
        """Return the temporal width after each consecutive grouping stage."""

        lengths: list[int] = []
        width = self.context_size
        for factor in self.group_factors:
            width //= factor
            lengths.append(width)
        return tuple(lengths)

    def fingerprint(self) -> str:
        """Return a stable digest of every architecture control."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()
