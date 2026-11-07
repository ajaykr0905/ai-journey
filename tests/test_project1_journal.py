from dataclasses import replace

import pytest

from ai_journey.project1_journal import Project1Event, chain_event, verify_event_chain


def event() -> Project1Event:
    return Project1Event(
        sequence=0,
        attempt_id="attempt-1-0123456789abcdef",
        kind="started",
        recorded_at_utc="2026-11-07T07:30:00+00:00",
        completed_step=0,
        payload={"host": "cpu", "parameters": 42},
    )


def test_event_preserves_canonical_content_safe_fields():
    assert event().to_mapping()["payload"] == {"host": "cpu", "parameters": 42}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sequence", -1, "non-negative"),
        ("attempt_id", "bad", "attempt_id"),
        ("kind", "unknown", "unsupported"),
        ("recorded_at_utc", "2026-11-07T07:30:00", "UTC"),
        ("completed_step", True, "integer"),
        ("payload", {"loss": float("nan")}, "finite"),
        ("payload", {"secret": object()}, "canonical JSON"),
    ],
)
def test_event_rejects_ambiguous_or_unbounded_values(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(event(), **{field: value})


def test_event_chain_binds_every_record_to_its_ancestry():
    first = chain_event(event())
    second_event = replace(event(), sequence=1, kind="heartbeat")
    second = chain_event(second_event, first["event_sha256"])

    assert verify_event_chain([first, second]) == (event(), second_event)


@pytest.mark.parametrize("tamper", ["payload", "ancestry", "sequence"])
def test_event_chain_rejects_tampering_and_gaps(tamper):
    first = chain_event(event())
    second_event = replace(event(), sequence=1, kind="heartbeat")
    second = chain_event(second_event, first["event_sha256"])
    records = [first, second]
    if tamper == "payload":
        records[0]["event"]["payload"]["parameters"] = 43
    elif tamper == "ancestry":
        records[1]["previous_sha256"] = "f" * 64
    else:
        changed = replace(second_event, sequence=2)
        records[1] = chain_event(changed, first["event_sha256"])

    with pytest.raises(ValueError, match="digest|ancestry|sequence"):
        verify_event_chain(records)
