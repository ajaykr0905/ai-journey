"""Exact metric events and summaries for Project 1 training."""

from __future__ import annotations

import math
from dataclasses import asdict

from .gpt2_training import GPT2StepMetric
from .project1_contract import Project1RunError
from .project1_journal import Project1Event


def step_metric_event(
    *,
    sequence: int,
    attempt_id: str,
    recorded_at_utc: str,
    metric: GPT2StepMetric,
) -> Project1Event:
    """Convert one measured optimizer update into a validated journal event."""

    if not isinstance(metric, GPT2StepMetric):
        raise TypeError("metric must be GPT2StepMetric")
    for name in (
        "mean_loss",
        "learning_rate",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
    ):
        if not math.isfinite(getattr(metric, name)):
            raise Project1RunError(f"metric {name} must be finite")
    if (
        metric.step <= 0
        or metric.mean_loss < 0
        or metric.learning_rate < 0
        or metric.gradient_norm_before_clip < 0
        or metric.gradient_norm_after_clip < 0
        or metric.micro_batches <= 0
        or metric.target_tokens <= 0
    ):
        raise Project1RunError("step metric counts and measurements are invalid")
    return Project1Event(
        sequence=sequence,
        attempt_id=attempt_id,
        kind="step",
        recorded_at_utc=recorded_at_utc,
        completed_step=metric.step,
        payload=asdict(metric),
    )
