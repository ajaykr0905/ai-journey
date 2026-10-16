"""Small, explicit low-rank adapters for the bounded CPU decoder lab."""

from dataclasses import dataclass
import math


class LoRAError(ValueError):
    """An adapter, its target, or its saved state violates the contract."""


@dataclass(frozen=True)
class LoRAConfig:
    """Predeclared adapter capacity, scaling, dropout, and local RNG seed."""

    rank: int = 2
    alpha: float = 4.0
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise LoRAError("rank must be an integer in [1, 256]")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or not 0 < self.alpha <= 1024
        ):
            raise LoRAError("alpha must be finite and in (0, 1024]")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise LoRAError("dropout must be finite and in [0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise LoRAError("seed must be an integer in [0, 2**32)")

    @property
    def scaling(self):
        return self.alpha / self.rank
