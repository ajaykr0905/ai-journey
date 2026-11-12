import copy

import pytest

from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from ai_journey.verifiable_reward_experiment import run_reward_experiment
from ai_journey.verifiable_reward_faults import FaultPlan
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_report import build_reward_report
from ai_journey.verifiable_reward_replay import replay_reward_report
from ai_journey.verifiable_reward_task import VerifiableTask


def test_replay_requires_every_report_byte_to_recompute():
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
    assert replay_reward_report(report, tasks, groups, config, faults)["byte_exact"]
    changed = copy.deepcopy(report)
    changed["evidence_boundary"] += "."
    with pytest.raises(VerifiableRewardError, match="byte exact"):
        replay_reward_report(changed, tasks, groups, config, faults)
