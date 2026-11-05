"""Predeclared controls for bounded scaling-law experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .scaling_laws import ScalingLawError, _positive_integer


@dataclass(frozen=True, order=True)
class ExperimentPoint:
    """One model/token/seed coordinate in a predeclared sweep."""

    parameter_count: int
    train_tokens: int
    seed: int

    def __post_init__(self) -> None:
        _positive_integer(self.parameter_count, "parameter_count")
        _positive_integer(self.train_tokens, "train_tokens")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ScalingLawError("seed must be nonnegative")
