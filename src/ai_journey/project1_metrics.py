"""Exact metric events and summaries for Project 1 training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .gpt2_training import GPT2StepMetric
from .project1_contract import Project1RunError
from .project1_journal import Project1Event


@dataclass(frozen=True)
class Project1MetricSummary:
    """Loss, exposure, and clipping totals derived from exact step events."""

    steps: int
    first_loss: float
    final_loss: float
    best_loss: float
    total_target_tokens: int
    clipped_steps: int
    maximum_gradient_norm: float


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


def summarize_step_events(events: Sequence[Project1Event]) -> Project1MetricSummary:
    """Summarize measured steps without inventing missing observations."""

    if not isinstance(events, Sequence) or not events:
        raise Project1RunError("step events must be a non-empty sequence")
    metrics: list[GPT2StepMetric] = []
    for event in events:
        if not isinstance(event, Project1Event) or event.kind != "step":
            raise Project1RunError("summary accepts only step events")
        try:
            metric = GPT2StepMetric(**event.payload)
        except TypeError as exc:
            raise Project1RunError("step event metric fields are invalid") from exc
        if metric.step != event.completed_step:
            raise Project1RunError("step event progress differs from its metric")
        step_metric_event(
            sequence=event.sequence,
            attempt_id=event.attempt_id,
            recorded_at_utc=event.recorded_at_utc,
            metric=metric,
        )
        metrics.append(metric)
    losses = [metric.mean_loss for metric in metrics]
    return Project1MetricSummary(
        steps=len(metrics),
        first_loss=losses[0],
        final_loss=losses[-1],
        best_loss=min(losses),
        total_target_tokens=sum(metric.target_tokens for metric in metrics),
        clipped_steps=sum(metric.clipped for metric in metrics),
        maximum_gradient_norm=max(
            metric.gradient_norm_before_clip for metric in metrics
        ),
    )
