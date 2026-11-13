import copy

import pytest

from ai_journey.recovery_trace import RecoveryEvent, chain_recovery_event


def event(**changes):
    values = {
        "sequence": 0,
        "run_id": "day-70-worker-0123456789abcdef",
        "attempt_id": "attempt-1-0123456789abcdef",
        "kind": "started",
        "monotonic_ns": 100,
        "completed_items": 0,
        "payload": {},
    }
    return RecoveryEvent(**(values | changes))


def test_event_mapping_is_canonical_and_chain_binds_prefix():
    first = chain_recovery_event(event())
    second = chain_recovery_event(
        event(sequence=1, kind="heartbeat", monotonic_ns=110),
        first["event_sha256"],
    )

    assert first["previous_sha256"] is None
    assert second["previous_sha256"] == first["event_sha256"]
    assert len(second["event_sha256"]) == 64
    changed = copy.deepcopy(first)
    changed["monotonic_ns"] = 101
    assert changed != first


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sequence", True, "integer"),
        ("monotonic_ns", -1, "non-negative"),
        ("completed_items", -1, "non-negative"),
        ("run_id", "bad", "run_id"),
        ("attempt_id", "bad", "attempt_id"),
        ("kind", "private-debug", "unsupported"),
        ("payload", {"bad": float("nan")}, "canonical JSON"),
    ],
)
def test_event_rejects_invalid_or_unbounded_evidence(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        event(**{field: value})


def test_chain_rejects_invalid_ancestry_digest():
    with pytest.raises(ValueError, match="SHA-256"):
        chain_recovery_event(event(), "bad")
