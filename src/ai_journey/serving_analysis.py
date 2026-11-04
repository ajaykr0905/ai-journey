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


@dataclass(frozen=True)
class GoodputTarget:
    """Bound first-token and end-to-end latency for useful completions."""

    max_ttft_ns: int
    max_end_to_end_ns: int

    def __post_init__(self) -> None:
        for name in ("max_ttft_ns", "max_end_to_end_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ServingAnalysisError(f"{name} must be positive")
        if self.max_ttft_ns > self.max_end_to_end_ns:
            raise ServingAnalysisError("TTFT target cannot exceed end-to-end target")


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


def goodput_fraction(
    observations: tuple[RequestObservation, ...], target: GoodputTarget
) -> Fraction:
    """Return the exact share of requests completing inside both SLOs."""

    require_unique_observation_ids(observations)
    if not isinstance(target, GoodputTarget):
        raise TypeError("target must be GoodputTarget")
    passing = 0
    for observation in observations:
        ttft = observation.timing.time_to_first_token_ns
        end_to_end = observation.timing.completed_ns - observation.timing.received_ns
        passing += int(
            observation.outcome is ServingOutcome.SUCCESS
            and ttft is not None
            and ttft <= target.max_ttft_ns
            and end_to_end <= target.max_end_to_end_ns
        )
    return Fraction(passing, len(observations))


def failure_fraction(
    observations: tuple[RequestObservation, ...],
) -> Fraction:
    """Return the exact share of non-success terminal outcomes."""

    require_unique_observation_ids(observations)
    failed = sum(
        observation.outcome is not ServingOutcome.SUCCESS
        for observation in observations
    )
    return Fraction(failed, len(observations))


def retry_amplification(
    observations: tuple[RequestObservation, ...],
) -> Fraction:
    """Return total attempts divided by unique scheduled requests."""

    require_unique_observation_ids(observations)
    attempts = len(observations) + sum(
        observation.retry_count for observation in observations
    )
    return Fraction(attempts, len(observations))
