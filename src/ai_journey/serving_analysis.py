"""Exact, deterministic analysis for bounded serving observations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .serving_observation import (
    RequestObservation,
    ServingOutcome,
    require_unique_observation_ids,
)


class ServingAnalysisError(ValueError):
    """Raised when serving metrics would be undefined or misleading."""


def nearest_rank_percentile(values: tuple[int, ...], percentile: int) -> int:
    """Return an integer nearest-rank percentile without interpolation."""

    if not isinstance(values, tuple) or not values:
        raise TypeError("values must be a nonempty integer tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ServingAnalysisError("percentile values must be nonnegative integers")
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, int)
        or not 1 <= percentile <= 100
    ):
        raise ServingAnalysisError("percentile must be an integer in [1, 100]")
    ordered = sorted(values)
    rank = (percentile * len(ordered) + 99) // 100
    return ordered[rank - 1]


@dataclass(frozen=True)
class LatencySummary:
    """Nearest-rank first-token latency percentiles in nanoseconds."""

    p50_ns: int
    p95_ns: int
    p99_ns: int


def successful_ttft_summary(
    observations: tuple[RequestObservation, ...],
) -> LatencySummary:
    """Summarize successful requests that emitted a first token."""

    require_unique_observation_ids(observations)
    values = tuple(
        observation.timing.time_to_first_token_ns
        for observation in observations
        if observation.outcome is ServingOutcome.SUCCESS
        and observation.timing.time_to_first_token_ns is not None
    )
    if not values:
        raise ServingAnalysisError("no successful first-token latency samples")
    return LatencySummary(
        p50_ns=nearest_rank_percentile(values, 50),
        p95_ns=nearest_rank_percentile(values, 95),
        p99_ns=nearest_rank_percentile(values, 99),
    )


def output_throughput_per_second(
    observations: tuple[RequestObservation, ...],
) -> Fraction:
    """Return successful generated tokens over the full observed wall interval."""

    require_unique_observation_ids(observations)
    duration_ns = max(
        observation.timing.completed_ns for observation in observations
    ) - min(observation.timing.received_ns for observation in observations)
    if duration_ns <= 0:
        raise ServingAnalysisError("throughput requires a positive observation span")
    generated = sum(
        observation.generated_tokens
        for observation in observations
        if observation.outcome is ServingOutcome.SUCCESS
    )
    return Fraction(generated * 1_000_000_000, duration_ns)
