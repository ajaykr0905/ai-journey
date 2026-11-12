"""Independent reconstruction and verification of sealed reward reports."""

from __future__ import annotations

from .verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from .verifiable_reward_experiment import run_reward_experiment
from .verifiable_reward_faults import FaultPlan
from .verifiable_reward_group import CandidateGroup
from .verifiable_reward_identity import sha256_json
from .verifiable_reward_report import build_reward_report
from .verifiable_reward_task import VerifiableTask


def verify_reward_report(
    report: dict[str, object],
    tasks: tuple[VerifiableTask, ...],
    groups: tuple[CandidateGroup, ...],
    config: RewardRunConfig,
    faults: FaultPlan,
) -> dict[str, object]:
    if set(report) != {
        "schema_version",
        "evidence_boundary",
        "config",
        "tasks",
        "groups",
        "faults",
        "result",
        "sha256",
    }:
        raise VerifiableRewardError("report fields mismatch")
    body = {key: report[key] for key in report if key != "sha256"}
    if report["sha256"] != sha256_json(body):
        raise VerifiableRewardError("report digest does not match")
    expected = build_reward_report(
        tasks,
        groups,
        config,
        faults,
        run_reward_experiment(tasks, groups, config, faults),
    )
    if report != expected:
        raise VerifiableRewardError(
            "report does not match independently reconstructed evidence"
        )
    return {
        "report_sha256": report["sha256"],
        "candidate_count": report["result"]["metrics"]["candidate_count"],
    }
