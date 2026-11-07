"""Failure and restart decisions for Project 1 training attempts."""

from __future__ import annotations

import math
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
