"""Canonical sealed reports for verifiable-reward experiments."""

from __future__ import annotations

from .verifiable_reward_contract import RewardRunConfig
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_group import CandidateGroup
from .verifiable_reward_identity import sha256_json
from .verifiable_reward_task import VerifiableTask


def build_reward_report(
    tasks: tuple[VerifiableTask, ...],
    groups: tuple[CandidateGroup, ...],
    config: RewardRunConfig,
    fault_plan: FaultPlan,
    result: dict[str, object],
) -> dict[str, object]:
    body = {
        "schema_version": 1,
        "evidence_boundary": "deterministic fixture responses; no model-quality, RL training, GPU, or production claim",
        "config": {
            "seed": config.seed,
            "candidates_per_task": config.candidates_per_task,
            "max_tasks": config.max_tasks,
            "max_response_bytes": config.max_response_bytes,
            "max_attempts": config.max_attempts,
        },
        "tasks": [task.to_mapping() for task in tasks],
        "groups": [
            {"task_id": group.task_id, "responses": list(group.responses)}
            for group in groups
        ],
        "faults": fault_plan.to_records(),
        "result": result,
    }
    return {**body, "sha256": sha256_json(body)}
