"""Composition of deterministic grouped reward evidence."""

from __future__ import annotations

from .verifiable_reward_budget import validate_run_budget
from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from .verifiable_reward_evaluator import evaluate_group
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_grpo import group_relative_advantages
from .verifiable_reward_identity import task_set_identity
from .verifiable_reward_journal import build_attempt_journal
from .verifiable_reward_metrics import summarize_attempts


def run_reward_experiment(
    tasks, groups, config: RewardRunConfig, fault_plan: FaultPlan
) -> dict[str, object]:
    tasks, groups = tuple(tasks), tuple(groups)
    validate_run_budget(task_count=len(tasks), group_count=len(groups), config=config)
    if [task.task_id for task in tasks] != [group.task_id for group in groups]:
        raise VerifiableRewardError("candidate group order must match task order")
    attempts = tuple(
        item
        for task, group in zip(tasks, groups, strict=True)
        for item in evaluate_group(task, group, config, fault_plan)
    )
    final = {}
    for item in attempts:
        final[(item.task_id, item.candidate_index)] = item
    advantages = {
        task.task_id: list(
            group_relative_advantages(
                [
                    final[(task.task_id, index)].reward
                    for index in range(config.candidates_per_task)
                ]
            )
        )
        for task in tasks
    }
    return {
        "task_set_sha256": task_set_identity(tasks),
        "attempts": [item.to_mapping() for item in attempts],
        "journal": build_attempt_journal(attempts),
        "advantages": advantages,
        "metrics": summarize_attempts(attempts),
    }
