from ai_journey.verifiable_reward_contract import RewardRunConfig
from ai_journey.verifiable_reward_evaluator import evaluate_group
from ai_journey.verifiable_reward_faults import FaultPlan, InjectedFault
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_task import VerifiableTask


def test_evaluator_retries_transient_fault_then_commits_verification():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "t",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    group = CandidateGroup("t", ("FINAL: ok", "FINAL: no"))
    config = RewardRunConfig(1, 2, 1, 100, 2)
    attempts = evaluate_group(
        task, group, config, FaultPlan([InjectedFault("t", 0, 1, "transient")])
    )

    assert [item.status for item in attempts] == [
        "transient_failure",
        "correct",
        "incorrect",
    ]
    assert [item.attempt for item in attempts] == [1, 2, 1]


def test_evaluator_does_not_retry_permanent_fault():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "t",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    attempts = evaluate_group(
        task,
        CandidateGroup("t", ("FINAL: ok", "FINAL: ok")),
        RewardRunConfig(1, 2, 1, 100, 3),
        FaultPlan([InjectedFault("t", 0, 1, "permanent")]),
    )
    assert [item.attempt for item in attempts if item.candidate_index == 0] == [1]
