import pytest

from ai_journey.verifiable_reward_answer import extract_final_answer
from ai_journey.verifiable_reward_contract import VerifiableRewardError


def test_extracts_one_explicit_final_answer_without_reasoning_text():
    assert extract_final_answer("work\nFINAL: 42\n", max_bytes=100) == "42"


@pytest.mark.parametrize("response", ["42", "FINAL:", "FINAL: a\nFINAL: b"])
def test_rejects_missing_empty_or_ambiguous_final_answer(response):
    with pytest.raises(VerifiableRewardError, match="exactly one"):
        extract_final_answer(response, max_bytes=100)


def test_enforces_utf8_byte_budget_not_character_count():
    with pytest.raises(VerifiableRewardError, match="byte budget"):
        extract_final_answer("FINAL: ₹", max_bytes=8)
