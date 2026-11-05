"""Transparent analysis for measured scaling experiment observations."""

from __future__ import annotations

from dataclasses import dataclass
import statistics

from .scaling_experiment_observation import ScalingObservation
from .scaling_laws import ScalingLawError


@dataclass(frozen=True, order=True)
class AggregatedPoint:
    parameter_count: int
    train_tokens: int
    mean_validation_loss: float
    sample_standard_deviation: float
    replicate_count: int


def aggregate_replicates(
    observations: tuple[ScalingObservation, ...],
) -> tuple[AggregatedPoint, ...]:
    """Aggregate seeds by coordinate without pooling different model/token pairs."""

    if not isinstance(observations, tuple) or not observations:
        raise ScalingLawError("observations must be a nonempty tuple")
    if any(not isinstance(item, ScalingObservation) for item in observations):
        raise TypeError("observations must contain ScalingObservation values")
    groups: dict[tuple[int, int], list[float]] = {}
    for item in observations:
        key = (item.point.parameter_count, item.point.train_tokens)
        groups.setdefault(key, []).append(float(item.validation_loss))
    if any(len(losses) < 2 for losses in groups.values()):
        raise ScalingLawError("every coordinate requires at least two replicates")
    return tuple(
        AggregatedPoint(
            parameter_count=key[0],
            train_tokens=key[1],
            mean_validation_loss=statistics.fmean(losses),
            sample_standard_deviation=statistics.stdev(losses),
            replicate_count=len(losses),
        )
        for key, losses in sorted(groups.items())
    )
