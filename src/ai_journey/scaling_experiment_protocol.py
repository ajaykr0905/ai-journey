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


def _strict_positive_axis(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or len(values) < 2:
        raise ScalingLawError(f"{name} must contain at least two values")
    for value in values:
        _positive_integer(value, name[:-1])
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ScalingLawError(f"{name} must be strictly increasing")
    return values


@dataclass(frozen=True)
class ScalingExperimentProtocol:
    """Immutable axes and seeds for one bounded scaling experiment."""

    name: str
    parameter_counts: tuple[int, ...]
    train_token_counts: tuple[int, ...]
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ScalingLawError("name must be nonempty")
        _strict_positive_axis(self.parameter_counts, "parameter_counts")
        _strict_positive_axis(self.train_token_counts, "train_token_counts")
        if not isinstance(self.seeds, tuple) or len(self.seeds) < 2:
            raise ScalingLawError("seeds must contain at least two values")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds
        ):
            raise TypeError("seeds must contain integers")
        if any(seed < 0 for seed in self.seeds):
            raise ScalingLawError("seeds must be nonnegative")
        if tuple(sorted(self.seeds)) != self.seeds or len(set(self.seeds)) != len(
            self.seeds
        ):
            raise ScalingLawError("seeds must be strictly increasing")


def build_experiment_grid(
    protocol: ScalingExperimentProtocol,
) -> tuple[ExperimentPoint, ...]:
    """Expand every declared coordinate in stable model/token/seed order."""

    if not isinstance(protocol, ScalingExperimentProtocol):
        raise TypeError("protocol must be ScalingExperimentProtocol")
    return tuple(
        ExperimentPoint(parameter_count, train_tokens, seed)
        for parameter_count in protocol.parameter_counts
        for train_tokens in protocol.train_token_counts
        for seed in protocol.seeds
    )
