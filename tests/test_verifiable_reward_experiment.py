import pytest

from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from ai_journey.verifiable_reward_experiment import run_reward_experiment
from ai_journey.verifiable_reward_faults import FaultPlan
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_journal import verify_attempt_journal
from ai_journey.verifiable_reward_task import VerifiableTask


def test_experiment_composes_attempts_advantages_metrics_and_journal():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "a",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    result = run_reward_experiment(
        (task,),
        (CandidateGroup("a", ("FINAL: ok", "FINAL: no")),),
        RewardRunConfig(1, 2, 1, 100, 1),
        FaultPlan(),
    )
    assert result["metrics"]["correct_count"] == 1
    assert result["advantages"]["a"][0] > result["advantages"]["a"][1]
    assert verify_attempt_journal(result["journal"])


def test_experiment_rejects_task_group_order_drift():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "a",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    with pytest.raises(VerifiableRewardError, match="order"):
        run_reward_experiment(
            (task,),
            (CandidateGroup("b", ("FINAL: ok", "FINAL: no")),),
            RewardRunConfig(1, 2, 1, 100, 1),
            FaultPlan(),
        )
