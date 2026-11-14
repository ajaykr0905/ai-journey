"""Client-observed recovery timing on the experiment host's monotonic clock.

Every timestamp comes from ``time.monotonic_ns`` in the experiment process: the
fault injector, the supervisor's exit and readiness observations, and each
client attempt. Server-process clocks are never mixed in. Durations are exact
integer nanoseconds. Latency percentiles use the nearest-rank method over
successful attempt latencies, sorted samples[ceil(p * n / 100) - 1].

* detection latency: kill issued -> exit status observed by the supervisor
* restart latency: exit observed -> replacement ready (includes the backoff)
* unavailability: first failed attempt finishing after the kill -> first
  success served by a post-fault lifetime (null when no client saw a failure
  before that success)
* time to SLO recovery: kill -> end of the first run of N consecutive in-SLO
  successes, in completion order; an in-SLO success started after the
  replacement was ready, was served by a post-fault lifetime and met the
  latency threshold. Any other attempt finishing in between resets the run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .serving_analysis import nearest_rank_percentile
from .serving_recovery_client import AttemptRecord
from .serving_recovery_config import ServingRecoveryError, require_int
from .serving_recovery_identity import lifetime_number

PERCENTILE_METHOD = (
    "nearest rank over successful attempt latencies: "
    "sorted samples[ceil(p*n/100)-1]; empty distributions are null"
)
_MAX_NS = 2**63 - 1


@dataclass(frozen=True)
class LatencyPercentiles:
    """Nearest-rank p50/p95/p99 of successful attempt latencies."""

    samples: int
    p50_ns: int | None
    p95_ns: int | None
    p99_ns: int | None


def latency_percentiles(values: Sequence[int]) -> LatencyPercentiles:
    """Summarize integer nanosecond latencies; empty inputs stay null."""

    values = tuple(values)
    if not values:
        return LatencyPercentiles(0, None, None, None)
    return LatencyPercentiles(
        len(values),
        nearest_rank_percentile(values, 50),
        nearest_rank_percentile(values, 95),
        nearest_rank_percentile(values, 99),
    )


@dataclass(frozen=True)
class ServingRecoveryTiming:
    """Measured recovery boundaries, durations and latency distributions."""

    fault_injected_ns: int
    exit_observed_ns: int
    replacement_ready_ns: int
    first_failure_after_fault_ns: int | None
    first_replacement_success_ns: int | None
    slo_recovered_ns: int | None
    detection_latency_ns: int
    restart_latency_ns: int
    unavailability_ns: int | None
    time_to_slo_recovery_ns: int | None
    late_acknowledgements: int
    pre_fault_latency: LatencyPercentiles
    post_fault_latency: LatencyPercentiles

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def measure_serving_recovery(
    *,
    attempts: Sequence[AttemptRecord],
    fault_injected_ns: int,
    faulted_lifetime_id: str,
    exit_observed_ns: int,
    replacement_lifetime_id: str,
    replacement_ready_ns: int,
    slo_latency_ms: int,
    slo_consecutive_successes: int,
) -> ServingRecoveryTiming:
    """Derive every recovery metric from monotonic attempt and lifecycle times."""

    for name, value in (
        ("fault_injected_ns", fault_injected_ns),
        ("exit_observed_ns", exit_observed_ns),
        ("replacement_ready_ns", replacement_ready_ns),
    ):
        require_int(value, name, 0, _MAX_NS)
    if not fault_injected_ns <= exit_observed_ns <= replacement_ready_ns:
        raise ServingRecoveryError("fault, exit and readiness must be ordered")
    faulted = lifetime_number(faulted_lifetime_id)
    if lifetime_number(replacement_lifetime_id) <= faulted:
        raise ServingRecoveryError("replacement must follow the faulted lifetime")
    slo_ns = require_int(slo_latency_ms, "slo_latency_ms", 1, 10_000) * 1_000_000
    needed = require_int(slo_consecutive_successes, "slo_consecutive_successes", 1, 64)
    if any(not isinstance(attempt, AttemptRecord) for attempt in attempts):
        raise TypeError("attempts must be AttemptRecord values")

    def post_fault(attempt: AttemptRecord) -> bool:
        return (
            attempt.outcome == "ok" and lifetime_number(attempt.lifetime_id) > faulted
        )

    after = sorted(
        (a for a in attempts if a.finished_ns > fault_injected_ns),
        key=lambda a: (a.finished_ns, a.request_index, a.attempt),
    )
    first_failure = min(
        (a.finished_ns for a in after if a.outcome != "ok"), default=None
    )
    first_success = min(
        (a.finished_ns for a in attempts if post_fault(a)), default=None
    )
    unavailability = None
    if None not in (first_failure, first_success) and first_failure <= first_success:
        unavailability = first_success - first_failure
    recovered_ns = None
    run = 0
    for attempt in after:
        in_slo = (
            post_fault(attempt)
            and attempt.started_ns >= replacement_ready_ns
            and attempt.latency_ns <= slo_ns
        )
        run = run + 1 if in_slo else 0
        if run == needed:
            recovered_ns = attempt.finished_ns
            break
    return ServingRecoveryTiming(
        fault_injected_ns=fault_injected_ns,
        exit_observed_ns=exit_observed_ns,
        replacement_ready_ns=replacement_ready_ns,
        first_failure_after_fault_ns=first_failure,
        first_replacement_success_ns=first_success,
        slo_recovered_ns=recovered_ns,
        detection_latency_ns=exit_observed_ns - fault_injected_ns,
        restart_latency_ns=replacement_ready_ns - exit_observed_ns,
        unavailability_ns=unavailability,
        time_to_slo_recovery_ns=(
            None if recovered_ns is None else recovered_ns - fault_injected_ns
        ),
        late_acknowledgements=sum(
            a.outcome == "ok" and not post_fault(a) for a in after
        ),
        pre_fault_latency=latency_percentiles(
            [
                a.latency_ns
                for a in attempts
                if a.outcome == "ok" and a.finished_ns <= fault_injected_ns
            ]
        ),
        post_fault_latency=latency_percentiles(
            [a.latency_ns for a in attempts if post_fault(a)]
        ),
    )
