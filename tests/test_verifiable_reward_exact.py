import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_exact import verify_exact


def test_exact_verifier_applies_declared_normalization_only():
    assert verify_exact("  EXACT restart ", "Exact Restart", {"case_sensitive": False})
    assert not verify_exact("exact  restart", "exact restart", {})
    assert verify_exact(
        "exact  restart", "exact restart", {"collapse_whitespace": True}
    )


def test_exact_verifier_rejects_unknown_or_mistyped_options():
    with pytest.raises(VerifiableRewardError, match="unsupported"):
        verify_exact("x", "x", {"fuzzy": True})
    with pytest.raises(VerifiableRewardError, match="booleans"):
        verify_exact("x", "x", {"case_sensitive": 1})
