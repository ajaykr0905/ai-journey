import copy

import pytest

from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from ai_journey.verifiable_reward_experiment import run_reward_experiment
from ai_journey.verifiable_reward_faults import FaultPlan
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_report import build_reward_report
from ai_journey.verifiable_reward_task import VerifiableTask
from ai_journey.verifiable_reward_verify import verify_reward_report


def fixture():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "a",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    tasks, groups, config, faults = (
        (task,),
        (CandidateGroup("a", ("FINAL: ok", "FINAL: no")),),
        RewardRunConfig(1, 2, 1, 100, 1),
        FaultPlan(),
    )
    report = build_reward_report(
        tasks,
        groups,
        config,
        faults,
        run_reward_experiment(tasks, groups, config, faults),
    )
    return report, tasks, groups, config, faults


def test_verifier_reconstructs_report_from_declared_inputs():
    report, tasks, groups, config, faults = fixture()
    assert (
        verify_reward_report(report, tasks, groups, config, faults)["candidate_count"]
        == 2
    )


def test_verifier_rejects_tampered_derived_metric():
    report, tasks, groups, config, faults = fixture()
    tampered = copy.deepcopy(report)
    tampered["result"]["metrics"]["correct_count"] = 2
    with pytest.raises(VerifiableRewardError, match="digest"):
        verify_reward_report(tampered, tasks, groups, config, faults)
