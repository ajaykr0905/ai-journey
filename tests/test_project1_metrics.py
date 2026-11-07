from dataclasses import replace

import pytest

from ai_journey.gpt2_training import GPT2StepMetric
from ai_journey.project1_metrics import step_metric_event


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
