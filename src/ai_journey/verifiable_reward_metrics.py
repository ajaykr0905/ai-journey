"""Reconciled metrics from final candidate attempts."""

from __future__ import annotations

import math

from .verifiable_reward_attempt import RewardAttempt


def summarize_attempts(attempts: tuple[RewardAttempt, ...]) -> dict[str, object]:
    final = {}
    for item in attempts:
        final[(item.task_id, item.candidate_index)] = item
    values = tuple(final.values())
    return {
        "attempt_count": len(attempts),
        "candidate_count": len(values),
        "retry_count": len(attempts) - len(values),
        "correct_count": sum(item.status == "correct" for item in values),
        "format_error_count": sum(item.status == "format_error" for item in values),
        "terminal_failure_count": sum(
            item.status.endswith("failure") for item in values
        ),
        "mean_reward": math.fsum(item.reward for item in values) / len(values)
        if values
        else 0.0,
    }
