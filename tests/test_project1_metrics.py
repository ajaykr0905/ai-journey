from dataclasses import replace

import pytest

from ai_journey.gpt2_training import GPT2StepMetric
from ai_journey.project1_metrics import step_metric_event, summarize_step_events


def metric() -> GPT2StepMetric:
    return GPT2StepMetric(
        step=1,
        mean_loss=4.2,
        learning_rate=3e-4,
        gradient_norm_before_clip=1.2,
        gradient_norm_after_clip=1.0,
        clipped=True,
        micro_batches=2,
        target_tokens=32,
    )


def test_step_metric_event_preserves_every_measured_training_field():
    event = step_metric_event(
        sequence=1,
        attempt_id="attempt-1-0123456789abcdef",
        recorded_at_utc="2026-11-07T07:31:00+00:00",
        metric=metric(),
    )

    assert event.completed_step == 1
    assert event.payload == {
        "step": 1,
        "mean_loss": 4.2,
        "learning_rate": 3e-4,
        "gradient_norm_before_clip": 1.2,
        "gradient_norm_after_clip": 1.0,
        "clipped": True,
        "micro_batches": 2,
        "target_tokens": 32,
    }


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"mean_loss": float("nan")}, "finite"),
        ({"learning_rate": -1.0}, "invalid"),
        ({"target_tokens": 0}, "invalid"),
    ],
)
def test_step_metric_event_rejects_nonfinite_or_impossible_metrics(changed, message):
    with pytest.raises(ValueError, match=message):
        step_metric_event(
            sequence=1,
            attempt_id="attempt-1-0123456789abcdef",
            recorded_at_utc="2026-11-07T07:31:00+00:00",
            metric=replace(metric(), **changed),
        )


def test_metric_summary_uses_exact_losses_tokens_and_clipping_observations():
    first = step_metric_event(
        sequence=1,
        attempt_id="attempt-1-0123456789abcdef",
        recorded_at_utc="2026-11-07T07:31:00+00:00",
        metric=metric(),
    )
    second = step_metric_event(
        sequence=2,
        attempt_id=first.attempt_id,
        recorded_at_utc="2026-11-07T07:32:00+00:00",
        metric=replace(
            metric(),
            step=2,
            mean_loss=3.8,
            clipped=False,
            gradient_norm_before_clip=0.9,
            gradient_norm_after_clip=0.9,
        ),
    )

    summary = summarize_step_events([first, second])
    assert summary.steps == 2
    assert summary.first_loss == 4.2
    assert summary.final_loss == summary.best_loss == 3.8
    assert summary.total_target_tokens == 64
    assert summary.clipped_steps == 1
    assert summary.maximum_gradient_norm == 1.2


def test_metric_summary_rejects_non_step_and_progress_mismatch():
    step = step_metric_event(
        sequence=1,
        attempt_id="attempt-1-0123456789abcdef",
        recorded_at_utc="2026-11-07T07:31:00+00:00",
        metric=metric(),
    )
    with pytest.raises(ValueError, match="only step"):
        summarize_step_events([replace(step, kind="heartbeat")])
    with pytest.raises(ValueError, match="progress"):
        summarize_step_events([replace(step, completed_step=2)])
