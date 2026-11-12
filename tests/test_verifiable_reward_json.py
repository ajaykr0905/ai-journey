import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_json import verify_json


def test_json_verifier_compares_structure_not_formatting():
    assert verify_json(
        '{"lost_steps":0,"retry":true}', {"retry": True, "lost_steps": 0}, {}
    )
    assert not verify_json("[0]", {"0": 0}, {})


def test_json_verifier_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(VerifiableRewardError, match="duplicate"):
        verify_json('{"x":1,"x":2}', {"x": 2}, {})
    with pytest.raises(VerifiableRewardError, match="strict finite"):
        verify_json('{"x":NaN}', {"x": 0}, {})
