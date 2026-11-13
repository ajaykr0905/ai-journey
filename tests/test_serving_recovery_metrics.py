import pytest
from test_serving_recovery_client import PLAN, RUN

from ai_journey.serving_recovery_client import AttemptRecord
from ai_journey.serving_recovery_identity import server_lifetime_id
from ai_journey.serving_recovery_metrics import (
    latency_percentiles,
    measure_serving_recovery,
)

MS = 1_000_000
FIRST = server_lifetime_id(RUN, 1)
SECOND = server_lifetime_id(RUN, 2)
THIRD = server_lifetime_id(RUN, 3)


def attempt(index, start_ms, latency_ms, outcome="ok", lifetime=FIRST, number=1):
    ok = outcome == "ok"
    return AttemptRecord(
        request_index=index,
        idempotency_key=PLAN[index].idempotency_key,
        attempt=number,
        port=8080,
        started_ns=start_ms * MS,
        finished_ns=(start_ms + latency_ms) * MS,
        outcome=outcome,
        error=None if ok else "scripted",
        http_status=200 if ok else None,
        lifetime_id=lifetime if ok else None,
        response_sha256="f" * 64 if ok else None,
    )


def timeline():
    """Kill at 1000 ms, exit seen at 1004 ms, replacement ready at 1100 ms."""

    return [
        attempt(0, 900, 10),
        attempt(1, 920, 20),
        attempt(2, 950, 40),
        attempt(3, 960, 30),
        attempt(4, 990, 15),
        attempt(5, 995, 7, "connection_reset"),
        attempt(6, 1010, 1, "connection_refused"),
        attempt(5, 1020, 1, "connection_refused", number=2),
        attempt(6, 1090, 5, lifetime=SECOND, number=2),
        attempt(5, 1101, 3, lifetime=SECOND, number=3),
        attempt(7, 1104, 250, lifetime=SECOND),
        attempt(8, 1110, 4, lifetime=SECOND),
        attempt(9, 1115, 6, lifetime=SECOND),
        attempt(10, 1121, 9, lifetime=SECOND),
    ]


def measure(attempts, **changes):
    options = {
        "attempts": attempts,
        "fault_injected_ns": 1000 * MS,
        "faulted_lifetime_id": FIRST,
        "exit_observed_ns": 1004 * MS,
        "replacement_lifetime_id": SECOND,
        "replacement_ready_ns": 1100 * MS,
        "slo_latency_ms": 100,
        "slo_consecutive_successes": 3,
    }
    return measure_serving_recovery(**(options | changes))


def test_recovery_timing_uses_only_monotonic_boundaries_and_attempts():
    timing = measure(timeline())

    assert timing.detection_latency_ns == 4 * MS
    assert timing.restart_latency_ns == 96 * MS
    assert timing.first_failure_after_fault_ns == 1002 * MS
    assert timing.first_replacement_success_ns == 1095 * MS
    assert timing.unavailability_ns == 93 * MS
    assert timing.slo_recovered_ns == 1121 * MS
    assert timing.time_to_slo_recovery_ns == 121 * MS
    assert timing.late_acknowledgements == 1
    assert timing.pre_fault_latency.samples == 4
    assert (
        timing.pre_fault_latency.p50_ns,
        timing.pre_fault_latency.p95_ns,
        timing.pre_fault_latency.p99_ns,
    ) == (20 * MS, 40 * MS, 40 * MS)
    assert timing.post_fault_latency.samples == 6
    assert timing.post_fault_latency.p50_ns == 5 * MS
    assert timing.post_fault_latency.p99_ns == 250 * MS
    assert timing.to_mapping()["pre_fault_latency"]["samples"] == 4


def test_slo_runs_follow_completion_order_and_reset_on_other_attempts():
    def recovered(attempts=None, **changes):
        return measure(attempts or timeline(), **changes).slo_recovered_ns

    assert recovered(slo_consecutive_successes=2) == 1114 * MS
    assert recovered(slo_consecutive_successes=4) == 1130 * MS
    assert recovered(slo_consecutive_successes=5) is None
    assert recovered(slo_consecutive_successes=5, slo_latency_ms=300) == 1354 * MS
    interrupted = timeline()
    interrupted.insert(-1, attempt(11, 1118, 2, "timeout"))
    assert recovered(interrupted) is None


def test_later_lifetimes_count_as_post_fault_and_gaps_stay_null():
    third = [attempt(12, 1200, 5, lifetime=THIRD)]
    timing = measure(timeline() + third, replacement_lifetime_id=THIRD)
    assert timing.post_fault_latency.samples == 7

    quiet = [a for a in timeline() if a.outcome == "ok"]
    assert measure(quiet).unavailability_ns is None
    nothing = measure(quiet[:5])
    assert nothing.first_replacement_success_ns is None
    assert nothing.post_fault_latency == latency_percentiles([])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"exit_observed_ns": 999 * MS}, "ordered"),
        ({"replacement_ready_ns": 1003 * MS}, "ordered"),
        ({"replacement_lifetime_id": FIRST}, "follow"),
        ({"slo_consecutive_successes": True}, "integer"),
        ({"slo_latency_ms": 0}, "slo_latency_ms"),
        ({"attempts": [object()]}, "AttemptRecord"),
    ],
)
def test_recovery_timing_rejects_incoherent_boundaries(changes, message):
    attempts = changes.pop("attempts", timeline())
    with pytest.raises((TypeError, ValueError), match=message):
        measure(attempts, **changes)
