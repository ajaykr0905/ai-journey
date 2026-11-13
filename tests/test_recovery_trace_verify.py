import copy

import pytest
from test_recovery_trace import event

from ai_journey.recovery_trace import chain_recovery_event
from ai_journey.recovery_trace_verify import verify_recovery_trace


def valid_records():
    specs = [
        {},
        {
            "sequence": 1,
            "kind": "item_completed",
            "monotonic_ns": 110,
            "completed_items": 1,
        },
        {
            "sequence": 2,
            "kind": "checkpoint",
            "monotonic_ns": 120,
            "completed_items": 1,
        },
        {
            "sequence": 3,
            "kind": "fault_injected",
            "monotonic_ns": 130,
            "completed_items": 2,
        },
        {
            "sequence": 4,
            "kind": "failure_detected",
            "monotonic_ns": 140,
            "completed_items": 2,
        },
        {
            "sequence": 5,
            "kind": "resumed",
            "attempt_id": "attempt-2-fedcba9876543210",
            "monotonic_ns": 150,
            "completed_items": 1,
        },
        {
            "sequence": 6,
            "kind": "completed",
            "attempt_id": "attempt-2-fedcba9876543210",
            "monotonic_ns": 160,
            "completed_items": 3,
        },
    ]
    records = []
    previous = None
    for values in specs:
        record = chain_recovery_event(event(**values), previous)
        records.append(record)
        previous = record["event_sha256"]
    return records


def test_verifier_accepts_one_explicit_rollback_and_attempt_change():
    events = verify_recovery_trace(valid_records())

    assert len(events) == 7
    assert events[5].kind == "resumed"
    assert events[-1].completed_items == 3


@pytest.mark.parametrize(
    ("index", "field", "value", "message"),
    [
        (1, "sequence", 3, "sequence"),
        (1, "previous_sha256", "0" * 64, "ancestry"),
        (1, "monotonic_ns", 99, "time moved backwards"),
        (1, "run_id", "other-run-0123456789abcdef", "run identity"),
        (5, "attempt_id", "attempt-1-0123456789abcdef", "new attempt"),
        (5, "completed_items", 3, "cannot exceed"),
        (6, "kind", "heartbeat", "end with completed"),
    ],
)
def test_verifier_rejects_corrupted_or_impossible_trace(index, field, value, message):
    records = copy.deepcopy(valid_records())
    records[index][field] = value
    if field not in {"sequence", "previous_sha256"}:
        previous = records[index]["previous_sha256"]
        event_fields = {
            key: item
            for key, item in records[index].items()
            if key not in {"previous_sha256", "event_sha256"}
        }
        records[index] = chain_recovery_event(event(**event_fields), previous)
        if index + 1 < len(records):
            records[index + 1]["previous_sha256"] = records[index]["event_sha256"]
    with pytest.raises(ValueError, match=message):
        verify_recovery_trace(records)


def test_verifier_rejects_unknown_record_fields():
    records = valid_records()
    records[0]["ignored"] = True
    with pytest.raises(ValueError, match="fields"):
        verify_recovery_trace(records)
