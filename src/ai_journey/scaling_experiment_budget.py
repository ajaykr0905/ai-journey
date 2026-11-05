"""Exact analytical compute accounting for scaling experiment grids."""

from __future__ import annotations

from dataclasses import dataclass

from .scaling_experiment_protocol import (
    ExperimentPoint,
    ScalingExperimentProtocol,
    build_experiment_grid,
)
from .scaling_laws import _positive_integer


@dataclass(frozen=True)
class ExperimentBudget:
    run_count: int
    total_training_flops: int
    maximum_run_flops: int


def point_training_flops(
    point: ExperimentPoint, *, flops_per_parameter_token: int = 6
) -> int:
    """Apply the declared dense 6ND-style convention to one point."""

    if not isinstance(point, ExperimentPoint):
        raise TypeError("point must be ExperimentPoint")
    convention = _positive_integer(
        flops_per_parameter_token, "flops_per_parameter_token"
    )
    return point.parameter_count * point.train_tokens * convention


def estimate_experiment_budget(
    protocol: ScalingExperimentProtocol, *, flops_per_parameter_token: int = 6
) -> ExperimentBudget:
    """Account for every seed at every declared model/token coordinate."""

    grid = build_experiment_grid(protocol)
    costs = tuple(
        point_training_flops(point, flops_per_parameter_token=flops_per_parameter_token)
        for point in grid
    )
    return ExperimentBudget(
        run_count=len(costs),
        total_training_flops=sum(costs),
        maximum_run_flops=max(costs),
    )
