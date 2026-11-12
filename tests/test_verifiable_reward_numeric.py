import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_numeric import verify_numeric


def test_numeric_verifier_uses_inclusive_decimal_tolerance():
    assert verify_numeric("0.30", "0.3", {})
    assert verify_numeric("9.99", 10, {"absolute_tolerance": "0.01"})
    assert not verify_numeric("9.98", 10, {"absolute_tolerance": "0.01"})


@pytest.mark.parametrize("answer", ["nan", "inf", "not-a-number"])
def test_numeric_verifier_rejects_nonfinite_or_invalid_answers(answer):
    with pytest.raises(VerifiableRewardError, match="answer"):
        verify_numeric(answer, 1, {})


def test_numeric_verifier_rejects_negative_tolerance():
    with pytest.raises(VerifiableRewardError, match="non-negative"):
        verify_numeric("1", 1, {"absolute_tolerance": "-1"})
