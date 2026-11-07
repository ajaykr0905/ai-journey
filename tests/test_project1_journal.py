from dataclasses import replace

import pytest

from ai_journey.project1_journal import Project1Event


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
