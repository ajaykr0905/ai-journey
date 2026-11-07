from dataclasses import replace

import pytest

from ai_journey.project1_journal import Project1Event
from ai_journey.project1_recovery import (
    failure_event,
    heartbeat_status,
    resume_decision,
)


def events():
    started = Project1Event(
        sequence=0,
        attempt_id="attempt-1-0123456789abcdef",
        kind="started",
        recorded_at_utc="2026-11-07T07:30:00+00:00",
        completed_step=0,
        payload={},
    )
    return [
        started,
        replace(
            started,
            sequence=1,
            kind="heartbeat",
            recorded_at_utc="2026-11-07T07:31:00+00:00",
        ),
    ]


def test_heartbeat_status_distinguishes_fresh_and_stale_attempts():
    fresh = heartbeat_status(
        events(),
        observed_at_utc="2026-11-07T07:31:30+00:00",
        timeout_seconds=60,
    )
    stale = heartbeat_status(
        events(),
        observed_at_utc="2026-11-07T07:32:01+00:00",
        timeout_seconds=60,
    )

    assert fresh.fresh is True and fresh.age_seconds == 30
    assert stale.fresh is False and stale.age_seconds == 61


def test_heartbeat_status_rejects_terminal_invalid_and_backward_observations():
    with pytest.raises(ValueError, match="running"):
        heartbeat_status(
            [*events(), replace(events()[-1], sequence=2, kind="failed")],
            observed_at_utc="2026-11-07T07:32:00+00:00",
            timeout_seconds=60,
        )
    with pytest.raises(ValueError, match="positive"):
        heartbeat_status(
            events(),
            observed_at_utc="2026-11-07T07:32:00+00:00",
            timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="precedes"):
        heartbeat_status(
            events(),
            observed_at_utc="2026-11-07T07:30:59+00:00",
            timeout_seconds=60,
        )


def test_failure_event_records_bounded_identity_without_exception_text():
    event = failure_event(
        sequence=2,
        attempt_id=events()[0].attempt_id,
        recorded_at_utc="2026-11-07T07:32:00+00:00",
        completed_step=0,
        error=RuntimeError("private path /tmp/example"),
        retryable=True,
    )

    assert event.kind == "failed"
    assert event.payload["error_type"] == "RuntimeError"
    assert event.payload["message_length"] == len("private path /tmp/example")
    assert event.payload["retryable"] is True
    assert "private path" not in str(event.payload)


def test_failure_event_requires_explicit_exception_and_retry_decision():
    arguments = {
        "sequence": 2,
        "attempt_id": events()[0].attempt_id,
        "recorded_at_utc": "2026-11-07T07:32:00+00:00",
        "completed_step": 0,
        "error": RuntimeError("boom"),
        "retryable": True,
    }
    with pytest.raises(TypeError, match="exception"):
        failure_event(**{**arguments, "error": "boom"})
    with pytest.raises(TypeError, match="boolean"):
        failure_event(**{**arguments, "retryable": 1})


def failed_events(*, retryable=True):
    started = events()[0]
    step = replace(
        started,
        sequence=1,
        kind="step",
        recorded_at_utc="2026-11-07T07:31:00+00:00",
        completed_step=1,
        payload={
            "step": 1,
            "mean_loss": 4.0,
            "learning_rate": 0.001,
            "gradient_norm_before_clip": 1.0,
            "gradient_norm_after_clip": 1.0,
            "clipped": False,
            "micro_batches": 1,
            "target_tokens": 8,
        },
    )
    checkpoint = replace(
        started,
        sequence=2,
        kind="checkpoint",
        recorded_at_utc="2026-11-07T07:32:00+00:00",
        completed_step=1,
    )
    failed = failure_event(
        sequence=3,
        attempt_id=started.attempt_id,
        recorded_at_utc="2026-11-07T07:33:00+00:00",
        completed_step=1,
        error=RuntimeError("interrupted"),
        retryable=retryable,
    )
    return [started, step, checkpoint, failed]


def test_resume_decision_binds_next_attempt_to_latest_checkpoint():
    decision = resume_decision(
        failed_events(),
        checkpoint_step=1,
        checkpoint_sha256="d" * 64,
        optimizer_steps=3,
    )

    assert decision.next_attempt_number == 2
    assert decision.checkpoint_step == 1
    assert decision.checkpoint_sha256 == "d" * 64


@pytest.mark.parametrize(
    ("retryable", "step", "digest", "optimizer_steps", "message"),
    [
        (False, 1, "d" * 64, 3, "retryable"),
        (True, 0, "d" * 64, 3, "journal progress"),
        (True, 1, "bad", 3, "SHA-256"),
        (True, 1, "d" * 64, 1, "schedule completion"),
    ],
)
def test_resume_decision_rejects_unproven_or_complete_lineage(
    retryable, step, digest, optimizer_steps, message
):
    with pytest.raises(ValueError, match=message):
        resume_decision(
            failed_events(retryable=retryable),
            checkpoint_step=step,
            checkpoint_sha256=digest,
            optimizer_steps=optimizer_steps,
        )
