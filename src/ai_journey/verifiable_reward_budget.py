"""Admission checks that prevent silent task or retry expansion."""

from __future__ import annotations

from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError


def validate_run_budget(
    *, task_count: int, group_count: int, config: RewardRunConfig
) -> int:
    if task_count < 1 or task_count > config.max_tasks:
        raise VerifiableRewardError("task count exceeds run contract")
    if group_count != task_count:
        raise VerifiableRewardError(
            "every admitted task requires exactly one candidate group"
        )
    return task_count * config.candidates_per_task * config.max_attempts
