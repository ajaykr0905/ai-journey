import pytest

from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from ai_journey.verifiable_reward_evaluator import evaluate_group
from ai_journey.verifiable_reward_faults import FaultPlan
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_recovery import make_checkpoint, resume_evaluation
from ai_journey.verifiable_reward_task import VerifiableTask


def test_resume_matches_uninterrupted_attempt_stream_exactly():
    tasks = tuple(
        VerifiableTask.from_mapping(
            {
                "task_id": x,
                "prompt": "p",
                "verifier": "exact",
                "expected": "ok",
                "options": {},
            }
        )
        for x in ("a", "b")
    )
    groups = tuple(CandidateGroup(x, ("FINAL: ok", "FINAL: no")) for x in ("a", "b"))
    config, faults = RewardRunConfig(1, 2, 2, 100, 2), FaultPlan()
    first = evaluate_group(tasks[0], groups[0], config, faults)
    resumed = resume_evaluation(
        tasks, groups, config, faults, make_checkpoint(tasks, 1, first), first
    )
    uninterrupted = first + evaluate_group(tasks[1], groups[1], config, faults)
    assert resumed == uninterrupted


def test_resume_rejects_modified_committed_attempts():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "a",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    config, group, faults = (
        RewardRunConfig(1, 2, 1, 100, 1),
        CandidateGroup("a", ("FINAL: ok", "FINAL: no")),
        FaultPlan(),
    )
    attempts = evaluate_group(task, group, config, faults)
    checkpoint = make_checkpoint((task,), 1, attempts)
    with pytest.raises(VerifiableRewardError, match="attempts"):
        resume_evaluation((task,), (group,), config, faults, checkpoint, attempts[:-1])
