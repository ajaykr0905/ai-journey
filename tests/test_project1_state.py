from dataclasses import replace

import pytest

from ai_journey.project1_journal import Project1Event
from ai_journey.project1_state import reduce_run_state


def events():
    base = Project1Event(
        sequence=0,
        attempt_id="attempt-1-0123456789abcdef",
        kind="started",
        recorded_at_utc="2026-11-07T07:30:00+00:00",
        completed_step=0,
        payload={},
    )
    return [
        base,
        replace(
            base,
            sequence=1,
            kind="step",
            completed_step=1,
            recorded_at_utc="2026-11-07T07:31:00+00:00",
        ),
        replace(
            base,
            sequence=2,
            kind="checkpoint",
            completed_step=1,
            recorded_at_utc="2026-11-07T07:32:00+00:00",
        ),
        replace(
            base,
            sequence=3,
            kind="completed",
            completed_step=1,
            recorded_at_utc="2026-11-07T07:33:00+00:00",
        ),
    ]


def test_lifecycle_reducer_derives_terminal_progress_and_checkpoint():
    state = reduce_run_state(events())

    assert state.status == "completed"
    assert state.completed_step == 1
    assert state.checkpoint_step == 1
    assert state.event_count == 4


@pytest.mark.parametrize("change", ["first", "gap", "time", "advance", "terminal"])
def test_lifecycle_reducer_rejects_impossible_transitions(change):
    changed = events()
    if change == "first":
        changed[0] = replace(changed[0], kind="heartbeat")
    elif change == "gap":
        changed[1] = replace(changed[1], completed_step=2)
    elif change == "time":
        changed[2] = replace(changed[2], recorded_at_utc="2026-11-07T07:29:00+00:00")
    elif change == "advance":
        changed[2] = replace(changed[2], completed_step=2)
    else:
        changed.insert(3, replace(changed[3], kind="failed"))
        changed[4] = replace(changed[4], sequence=4)

    with pytest.raises(ValueError):
        reduce_run_state(changed)
