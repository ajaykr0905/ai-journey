import copy

import pytest

from ai_journey.serving_recovery_identity import server_lifetime_id
from ai_journey.serving_recovery_trace import (
    ServingLifecycleEvent,
    chain_serving_trace,
    verify_serving_trace,
)

RUN = "serving-recovery-test-0123456789abcdef"
L = {number: server_lifetime_id(RUN, number) for number in (1, 2, 3)}
PID = {1: 101, 2: 202, 3: 303}


def step(kind, number, **payload):
    defaults = {
        "spawn": {"pid": PID[number]},
        "ready": {"pid": PID[number], "port": 8000 + number},
        "unresponsive": {"pid": PID[number]},
        "fault": {"pid": PID[number], "signal": "SIGKILL", "acknowledged_responses": 6},
        "exit_observed": {"pid": PID[number], "exit_code": -9},
        "restart": {
            "pid": PID[number],
            "parent_lifetime_id": L.get(number - 1, L[1]),
            "backoff_ms": 20,
        },
        "exhausted": {"restarts": 2},
        "recovered": {"consecutive_successes": 4, "slo_latency_ms": 200},
        "finished": {"pid": PID[number], "exit_code": 0},
    }[kind]
    return kind, number, defaults | payload


def events(*steps):
    return [
        ServingLifecycleEvent(sequence, RUN, L[number], kind, 1000 + sequence, payload)
        for sequence, (kind, number, payload) in enumerate(steps)
    ]


HAPPY = (
    step("spawn", 1),
    step("ready", 1),
    step("fault", 1),
    step("exit_observed", 1),
    step("restart", 2),
    step("ready", 2),
    step("recovered", 2),
    step("finished", 2),
)


def test_complete_lifecycle_chains_and_verifies():
    records = chain_serving_trace(events(*HAPPY))
    verified = verify_serving_trace(records)

    assert [event.kind for event in verified] == [kind for kind, _, _ in HAPPY]
    assert records[0]["previous_sha256"] is None
    assert all(
        later["previous_sha256"] == earlier["event_sha256"]
        for earlier, later in zip(records, records[1:])
    )
    assert len({record["event_sha256"] for record in records}) == len(records)


def test_failed_replacements_and_exhaustion_are_valid_histories():
    retried = HAPPY[:4] + (
        step("restart", 2),
        step("exit_observed", 2, exit_code=4),
        step("restart", 3, backoff_ms=40),
        step("ready", 3),
        step("recovered", 3),
        step("finished", 3),
    )
    verify_serving_trace(chain_serving_trace(events(*retried)))
    exhausted = HAPPY[:4] + (
        step("restart", 2),
        step("exit_observed", 2, exit_code=4),
        step("exhausted", 2),
    )
    records = chain_serving_trace(events(*exhausted))
    with pytest.raises(ValueError, match="one fault and finishes"):
        verify_serving_trace(records)
    unrecovered = HAPPY[:6] + (step("finished", 2),)
    verify_serving_trace(chain_serving_trace(events(*unrecovered)))


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ((step("ready", 1),), "must begin"),
        ((step("spawn", 2),), "first lifetime"),
        ((step("spawn", 1), step("fault", 1)), "cannot follow spawned"),
        (HAPPY[:2] + (step("restart", 2),), "observed exit"),
        (HAPPY[:4] + (step("restart", 3, parent_lifetime_id=L[2]),), "parent"),
        (HAPPY[:4] + (step("restart", 3, parent_lifetime_id=L[1]),), "skips"),
        (HAPPY[:2] + (step("recovered", 1),), "exactly one fault"),
        (HAPPY[:3] + (step("recovered", 1),), "cannot follow faulted"),
        (HAPPY[:6] + (step("fault", 2),), "more than one fault"),
        (HAPPY + (step("ready", 2),), "terminal"),
        (HAPPY[:2] + (step("exit_observed", 1, pid=999),), "another process"),
        (HAPPY[:2] + (step("ready", 2),), "another lifetime"),
        (HAPPY[:2] + (step("fault", 1, signal="SIGTERM"),), "SIGKILL"),
        (HAPPY[:3] + (step("exit_observed", 1, exit_code=0),), "exit by SIGKILL"),
        (HAPPY[:5] + (step("finished", 2),), "cannot follow spawned"),
    ],
)
def test_impossible_lifecycle_orders_are_rejected(steps, message):
    with pytest.raises(ValueError, match=message):
        chain_serving_trace(events(*steps))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"pid": True}, "wrong type"),
        ({"pid": 1, "extra": 2}, "schema"),
        ({}, "schema"),
    ],
)
def test_event_payloads_have_one_exact_schema(payload, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ServingLifecycleEvent(0, RUN, L[1], "spawn", 1, payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r[3]["payload"].update(exit_code=0), "digest"),
        (lambda r: r.__setitem__(4, r[5]), "contiguous"),
        (lambda r: r[2].update(previous_sha256="0" * 64), "ancestry"),
        (lambda r: r[1].update(monotonic_ns=1), "digest"),
        (lambda r: r[0].update(extra=True), "fields"),
        (lambda r: r.clear(), "empty"),
    ],
)
def test_tampered_records_fail_verification(mutate, message):
    records = copy.deepcopy(chain_serving_trace(events(*HAPPY)))
    mutate(records)
    with pytest.raises(ValueError, match=message):
        verify_serving_trace(records)


def test_time_and_run_identity_are_bound_across_the_chain():
    backwards = events(*HAPPY)
    backwards[2] = ServingLifecycleEvent(2, RUN, L[1], "fault", 1, HAPPY[2][2])
    with pytest.raises(ValueError, match="backwards"):
        chain_serving_trace(backwards)
    other_run = events(*HAPPY)
    other_run[1] = ServingLifecycleEvent(
        1, "other-run-0123456789abcdef", L[1], "ready", 1001, HAPPY[1][2]
    )
    with pytest.raises(ValueError, match="run identity"):
        chain_serving_trace(other_run)
