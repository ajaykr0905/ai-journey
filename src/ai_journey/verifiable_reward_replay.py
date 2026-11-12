"""Byte-exact replay of a sealed reward experiment."""

from __future__ import annotations

from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from .verifiable_reward_experiment import run_reward_experiment
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_group import CandidateGroup
from .verifiable_reward_identity import canonical_json_bytes
from .verifiable_reward_report import build_reward_report
from .verifiable_reward_task import VerifiableTask


def replay_reward_report(
    report: dict[str, object],
    tasks: tuple[VerifiableTask, ...],
    groups: tuple[CandidateGroup, ...],
    config: RewardRunConfig,
    faults: FaultPlan,
) -> dict[str, object]:
    replayed = build_reward_report(
        tasks,
        groups,
        config,
        faults,
        run_reward_experiment(tasks, groups, config, faults),
    )
    if canonical_json_bytes(report) != canonical_json_bytes(replayed):
        raise VerifiableRewardError("reward report replay is not byte exact")
    return {"report_sha256": replayed["sha256"], "byte_exact": True}
