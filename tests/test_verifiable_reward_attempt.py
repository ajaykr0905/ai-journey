import pytest

from ai_journey.verifiable_reward_attempt import RewardAttempt
from ai_journey.verifiable_reward_contract import VerifiableRewardError


def test_attempt_serializes_complete_audit_record():
    attempt = RewardAttempt("task", 0, 1, "correct", 1.0, "correct")
    assert attempt.to_mapping()["candidate_index"] == 0


def test_attempt_rejects_rewarded_failure_and_out_of_range_reward():
    with pytest.raises(VerifiableRewardError, match="cannot receive"):
        RewardAttempt("task", 0, 1, "transient_failure", 0.2, "injected")
    with pytest.raises(VerifiableRewardError, match=r"\[0, 1\]"):
        RewardAttempt("task", 0, 1, "correct", 1.1, "bad")
