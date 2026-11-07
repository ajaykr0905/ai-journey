"""Lifecycle reduction for verified Project 1 event journals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from .project1_contract import Project1RunError
from .project1_journal import Project1Event


@dataclass(frozen=True)
class Project1RunState:
    """Current durable state derived only from ordered journal evidence."""

    status: str
    attempt_id: str
    completed_step: int
    checkpoint_step: int | None
    event_count: int


def reduce_run_state(events: Sequence[Project1Event]) -> Project1RunState:
    """Reject impossible transitions and derive the current attempt state."""

    if not isinstance(events, Sequence) or not events:
        raise Project1RunError("events must be a non-empty sequence")
    if any(not isinstance(event, Project1Event) for event in events):
        raise TypeError("events must contain Project1Event values")
    first = events[0]
    if first.sequence != 0 or first.kind != "started":
        raise Project1RunError("attempt journal must begin with started")
    attempt, completed, checkpoint = first.attempt_id, first.completed_step, None
    previous_time = datetime.fromisoformat(first.recorded_at_utc)
    terminal = False
    status = "running"
    for index, event in enumerate(events):
        timestamp = datetime.fromisoformat(event.recorded_at_utc)
        if event.sequence != index or event.attempt_id != attempt:
            raise Project1RunError("journal sequence or attempt identity changed")
        if timestamp < previous_time:
            raise Project1RunError("journal timestamps must be non-decreasing")
        if event.completed_step < completed:
            raise Project1RunError("completed_step cannot move backwards")
        if event.kind == "step" and event.completed_step != completed + 1:
            raise Project1RunError(
                "step events must advance exactly one optimizer step"
            )
        if event.kind != "step" and event.completed_step != completed:
            raise Project1RunError("non-step events cannot advance optimizer progress")
        if terminal and index:
            raise Project1RunError("terminal events must end the journal")
        if event.kind == "checkpoint":
            checkpoint = event.completed_step
        if event.kind in {"failed", "completed"}:
            status, terminal = event.kind, True
        completed, previous_time = event.completed_step, timestamp
    return Project1RunState(
        status=status,
        attempt_id=attempt,
        completed_step=completed,
        checkpoint_step=checkpoint,
        event_count=len(events),
    )
