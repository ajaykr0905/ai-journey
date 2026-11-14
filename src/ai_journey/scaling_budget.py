"""Invert declared training-compute budgets and compare IsoFLOP allocations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from .scaling_laws import (
    ComputeOptimalTarget,
    ScalingLawError,
    _positive_decimal,
    _positive_integer,
)


def token_capacity(
    *,
    compute_budget_flops: int,
    parameter_count: int,
    training_flops_per_parameter_token: int | float | str = "6",
) -> int:
    """Return the maximum whole tokens admitted by an analytical budget."""

    budget = _positive_integer(compute_budget_flops, "compute_budget_flops")
    parameters = _positive_integer(parameter_count, "parameter_count")
    convention = _positive_decimal(
        training_flops_per_parameter_token,
        "training_flops_per_parameter_token",
    )
    capacity = Decimal(budget) / (Decimal(parameters) * convention)
    return int(capacity.to_integral_value(rounding=ROUND_FLOOR))


def parameter_capacity(
    *,
    compute_budget_flops: int,
    token_count: int,
    training_flops_per_parameter_token: int | float | str = "6",
) -> int:
    """Return the maximum whole parameters admitted by an analytical budget."""

    budget = _positive_integer(compute_budget_flops, "compute_budget_flops")
    tokens = _positive_integer(token_count, "token_count")
    convention = _positive_decimal(
        training_flops_per_parameter_token,
        "training_flops_per_parameter_token",
    )
    capacity = Decimal(budget) / (Decimal(tokens) * convention)
    return int(capacity.to_integral_value(rounding=ROUND_FLOOR))


@dataclass(frozen=True)
class IsoFLOPCandidate:
    """One model size given the whole budget, and its distance from the target ratio.

    ``absolute_ratio_error`` ranks rows in closest_compute_optimal_candidate,
    and training_flops never exceeds the budget.
    """

    parameter_count: int
    token_count: int
    training_flops: int
    tokens_per_parameter: Decimal
    optimal_tokens_per_parameter: Decimal
    absolute_ratio_error: Decimal


@dataclass(frozen=True)
class ComputeOptimalCapacity:
    """The largest whole model a budget funds at the declared ratio, and leftovers."""

    parameter_count: int
    token_count: int
    training_flops: int
    unused_flops: int


def compute_optimal_capacity(
    *,
    compute_budget_flops: int,
    tokens_per_parameter: int | float | str = "20",
    training_flops_per_parameter_token: int | float | str = "6",
) -> ComputeOptimalCapacity:
    """Find the largest whole model at the declared token ratio within a budget."""

    budget = _positive_integer(compute_budget_flops, "compute_budget_flops")
    ratio = _positive_decimal(tokens_per_parameter, "tokens_per_parameter")
    convention = _positive_decimal(
        training_flops_per_parameter_token,
        "training_flops_per_parameter_token",
    )
    parameters = int(
        (Decimal(budget) / (ratio * convention))
        .sqrt()
        .to_integral_value(rounding=ROUND_FLOOR)
    )
    if parameters < 1:
        raise ScalingLawError("compute budget cannot fund one parameter at the ratio")
    while parameters:
        target = ComputeOptimalTarget(
            parameters,
            tokens_per_parameter=tokens_per_parameter,
            training_flops_per_parameter_token=training_flops_per_parameter_token,
        )
        if target.training_flops <= budget:
            return ComputeOptimalCapacity(
                parameter_count=parameters,
                token_count=target.token_target,
                training_flops=target.training_flops,
                unused_flops=budget - target.training_flops,
            )
        parameters -= 1
    raise ScalingLawError("compute budget cannot fund one parameter at the ratio")


def build_isoflop_sweep(
    *,
    compute_budget_flops: int,
    parameter_counts: tuple[int, ...],
    optimal_tokens_per_parameter: int | float | str = "20",
    training_flops_per_parameter_token: int | float | str = "6",
) -> tuple[IsoFLOPCandidate, ...]:
    """Allocate one compute budget across declared model sizes.

    Each row consumes at most the budget. Rows that cannot train on even one
    token are rejected rather than silently reported as valid experiments.
    """

    budget = _positive_integer(compute_budget_flops, "compute_budget_flops")
    if not isinstance(parameter_counts, tuple) or not parameter_counts:
        raise ScalingLawError("parameter_counts must be a nonempty tuple")
    optimal = _positive_decimal(
        optimal_tokens_per_parameter, "optimal_tokens_per_parameter"
    )
    if len(set(parameter_counts)) != len(parameter_counts):
        raise ScalingLawError("parameter_counts must be unique")
    candidates = []
    for parameters in sorted(parameter_counts):
        _positive_integer(parameters, "parameter_count")
        tokens = token_capacity(
            compute_budget_flops=budget,
            parameter_count=parameters,
            training_flops_per_parameter_token=training_flops_per_parameter_token,
        )
        if tokens < 1:
            raise ScalingLawError("compute budget cannot fund every candidate")
        target = ComputeOptimalTarget(
            parameters,
            tokens_per_parameter=optimal_tokens_per_parameter,
            training_flops_per_parameter_token=training_flops_per_parameter_token,
        )
        actual_ratio = Decimal(tokens) / Decimal(parameters)
        candidates.append(
            IsoFLOPCandidate(
                parameter_count=parameters,
                token_count=tokens,
                training_flops=(
                    Decimal(parameters)
                    * Decimal(tokens)
                    * _positive_decimal(
                        training_flops_per_parameter_token,
                        "training_flops_per_parameter_token",
                    )
                ).__ceil__(),
                tokens_per_parameter=actual_ratio,
                optimal_tokens_per_parameter=optimal,
                absolute_ratio_error=abs(actual_ratio - optimal),
            )
        )
        if candidates[-1].training_flops > budget:
            raise AssertionError("IsoFLOP candidate exceeds its budget")
        if target.parameter_count != parameters:
            raise AssertionError("candidate parameter identity changed")
    return tuple(candidates)


def closest_compute_optimal_candidate(
    candidates: tuple[IsoFLOPCandidate, ...],
) -> IsoFLOPCandidate:
    """Select the smallest-ratio-error row with deterministic tie breaking."""

    if not isinstance(candidates, tuple) or not candidates:
        raise ScalingLawError("candidates must be a nonempty tuple")
    if any(not isinstance(row, IsoFLOPCandidate) for row in candidates):
        raise TypeError("candidates must contain IsoFLOPCandidate rows")
    return min(
        candidates, key=lambda row: (row.absolute_ratio_error, row.parameter_count)
    )
