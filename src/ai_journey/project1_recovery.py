"""Failure and restart decisions for Project 1 training attempts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from .project1_contract import Project1RunError
from .project1_journal import Project1Event
from .project1_state import reduce_run_state


@dataclass(frozen=True)
class Project1HeartbeatStatus:
    """Freshness of the latest durable observation for a running attempt."""

    fresh: bool
    age_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class Project1ResumeDecision:
    """Checkpoint lineage admitted for the next training attempt."""

    next_attempt_number: int
    checkpoint_step: int
    checkpoint_sha256: str


def heartbeat_status(
    events: Sequence[Project1Event],
    *,
    observed_at_utc: str,
    timeout_seconds: float,
) -> Project1HeartbeatStatus:
    """Classify an unfinished attempt without relying on process-local state."""

    state = reduce_run_state(events)
    if state.status != "running":
        raise Project1RunError("heartbeat freshness applies only to running attempts")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise Project1RunError("timeout_seconds must be finite and positive")
    try:
        observed = datetime.fromisoformat(observed_at_utc)
    except (TypeError, ValueError) as exc:
        raise Project1RunError("observed_at_utc must be an ISO-8601 timestamp") from exc
    latest = datetime.fromisoformat(events[-1].recorded_at_utc)
    if observed.tzinfo is None or observed.utcoffset().total_seconds() != 0:
        raise Project1RunError("observed_at_utc must use UTC")
    age = (observed - latest).total_seconds()
    if age < 0:
        raise Project1RunError("observation time precedes the latest event")
    return Project1HeartbeatStatus(
        fresh=age <= timeout_seconds,
        age_seconds=age,
        timeout_seconds=float(timeout_seconds),
    )


def failure_event(
    *,
    sequence: int,
    attempt_id: str,
    recorded_at_utc: str,
    completed_step: int,
    error: BaseException,
    retryable: bool,
) -> Project1Event:
    """Record failure identity without copying exception text into public evidence."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if not isinstance(retryable, bool):
        raise TypeError("retryable must be a boolean")
    message = str(error)
    return Project1Event(
        sequence=sequence,
        attempt_id=attempt_id,
        kind="failed",
        recorded_at_utc=recorded_at_utc,
        completed_step=completed_step,
        payload={
            "error_type": type(error).__name__,
            "message_sha256": sha256(message.encode()).hexdigest(),
            "message_length": len(message),
            "retryable": retryable,
        },
    )


def resume_decision(
    events: Sequence[Project1Event],
    *,
    checkpoint_step: int,
    checkpoint_sha256: str,
    optimizer_steps: int,
) -> Project1ResumeDecision:
    """Admit retry only from a retryable failure and its latest checkpoint."""

    state = reduce_run_state(events)
    if state.status != "failed" or events[-1].payload.get("retryable") is not True:
        raise Project1RunError("resume requires an explicitly retryable failed attempt")
    for name, value in (
        ("checkpoint_step", checkpoint_step),
        ("optimizer_steps", optimizer_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if (
        state.checkpoint_step is None
        or checkpoint_step != state.checkpoint_step
        or checkpoint_step > state.completed_step
    ):
        raise Project1RunError("resume checkpoint does not match journal progress")
    if checkpoint_step >= optimizer_steps:
        raise Project1RunError("resume checkpoint must precede schedule completion")
    if (
        not isinstance(checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
    ):
        raise Project1RunError("checkpoint_sha256 must be a SHA-256 digest")
    attempt_number = int(state.attempt_id.split("-", 2)[1])
    return Project1ResumeDecision(
        next_attempt_number=attempt_number + 1,
        checkpoint_step=checkpoint_step,
        checkpoint_sha256=checkpoint_sha256,
    )
