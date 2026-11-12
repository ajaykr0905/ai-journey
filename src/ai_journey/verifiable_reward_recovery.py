"""Exact group-boundary restart for reward evaluation."""

from __future__ import annotations

from .verifiable_reward_attempt import RewardAttempt
from .verifiable_reward_checkpoint import RewardCheckpoint
from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from .verifiable_reward_evaluator import evaluate_group
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_group import CandidateGroup
from .verifiable_reward_identity import sha256_json, task_set_identity
from .verifiable_reward_task import VerifiableTask


def make_checkpoint(
    tasks, next_group_index: int, attempts: tuple[RewardAttempt, ...]
) -> RewardCheckpoint:
    if not 0 <= next_group_index <= len(tasks):
        raise VerifiableRewardError("checkpoint group boundary is outside task set")
    return RewardCheckpoint(
        task_set_identity(tasks),
        next_group_index,
        sha256_json([item.to_mapping() for item in attempts]),
    )


def resume_evaluation(
    tasks: tuple[VerifiableTask, ...],
    groups: tuple[CandidateGroup, ...],
    config: RewardRunConfig,
    fault_plan: FaultPlan,
    checkpoint: RewardCheckpoint,
    committed_attempts: tuple[RewardAttempt, ...],
) -> tuple[RewardAttempt, ...]:
    if checkpoint.task_set_sha256 != task_set_identity(tasks):
        raise VerifiableRewardError("checkpoint task set does not match")
    if checkpoint.attempts_sha256 != sha256_json(
        [item.to_mapping() for item in committed_attempts]
    ):
        raise VerifiableRewardError("checkpoint attempts do not match")
    attempts = list(committed_attempts)
    for task, group in zip(
        tasks[checkpoint.next_group_index :],
        groups[checkpoint.next_group_index :],
        strict=True,
    ):
        attempts.extend(evaluate_group(task, group, config, fault_plan))
    return tuple(attempts)
