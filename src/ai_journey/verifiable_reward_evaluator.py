"""Retry-bounded evaluation of one candidate group."""

from __future__ import annotations

from .verifiable_reward_attempt import RewardAttempt
from .verifiable_reward_contract import RewardRunConfig
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_group import CandidateGroup
from .verifiable_reward_registry import verify_response
from .verifiable_reward_score import score_outcome
from .verifiable_reward_task import VerifiableTask


def evaluate_group(
    task: VerifiableTask,
    group: CandidateGroup,
    config: RewardRunConfig,
    fault_plan: FaultPlan,
) -> tuple[RewardAttempt, ...]:
    attempts: list[RewardAttempt] = []
    for candidate_index, response in enumerate(group.responses):
        for attempt_number in range(1, config.max_attempts + 1):
            fault = fault_plan.kind_for(task.task_id, candidate_index, attempt_number)
            if fault:
                status = f"{fault}_failure"
                attempts.append(
                    RewardAttempt(
                        task.task_id,
                        candidate_index,
                        attempt_number,
                        status,
                        0.0,
                        "injected fault",
                    )
                )
                if fault == "transient" and attempt_number < config.max_attempts:
                    continue
                break
            outcome = verify_response(
                task, response, max_bytes=config.max_response_bytes
            )
            status = (
                "correct"
                if outcome.correct
                else "incorrect"
                if outcome.format_valid
                else "format_error"
            )
            attempts.append(
                RewardAttempt(
                    task.task_id,
                    candidate_index,
                    attempt_number,
                    status,
                    score_outcome(outcome),
                    outcome.message,
                )
            )
            break
    return tuple(attempts)
