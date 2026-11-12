import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_registry import VerificationOutcome
from ai_journey.verifiable_reward_score import RewardWeights, score_outcome


def test_score_distinguishes_format_from_correctness():
    assert score_outcome(VerificationOutcome(False, False, "bad")) == 0.0
    assert score_outcome(VerificationOutcome(True, False, "wrong")) == 0.2
    assert score_outcome(VerificationOutcome(True, True, "correct")) == 1.0


def test_weights_must_form_a_probability_mass():
    with pytest.raises(VerifiableRewardError, match="sum to one"):
        RewardWeights(0.5, 0.6)
