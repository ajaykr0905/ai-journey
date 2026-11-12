from ai_journey.verifiable_reward_contract import RewardRunConfig
from ai_journey.verifiable_reward_faults import FaultPlan
from ai_journey.verifiable_reward_group import CandidateGroup
from ai_journey.verifiable_reward_identity import sha256_json
from ai_journey.verifiable_reward_report import build_reward_report
from ai_journey.verifiable_reward_task import VerifiableTask


def test_report_seals_inputs_results_and_honest_evidence_boundary():
    task = VerifiableTask.from_mapping(
        {
            "task_id": "a",
            "prompt": "p",
            "verifier": "exact",
            "expected": "ok",
            "options": {},
        }
    )
    report = build_reward_report(
        (task,),
        (CandidateGroup("a", ("FINAL: ok", "FINAL: no")),),
        RewardRunConfig(1, 2, 1, 100, 1),
        FaultPlan(),
        {"metrics": {}},
    )
    body = {key: value for key, value in report.items() if key != "sha256"}
    assert report["sha256"] == sha256_json(body)
    assert "no model-quality" in report["evidence_boundary"]
