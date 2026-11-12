import pytest

from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError
from ai_journey.verifiable_reward_group import admit_candidate_group


CONFIG = RewardRunConfig(1, 2, 3, 16, 2)


def test_group_preserves_candidate_order_and_duplicates():
    group = admit_candidate_group("task", ["FINAL: x", "FINAL: x"], CONFIG)
    assert group.responses == ("FINAL: x", "FINAL: x")


def test_group_enforces_declared_cardinality_and_bytes():
    with pytest.raises(VerifiableRewardError, match="count"):
        admit_candidate_group("task", ["FINAL: x"], CONFIG)
    with pytest.raises(VerifiableRewardError, match="byte budget"):
        admit_candidate_group("task", ["x" * 17, "y"], CONFIG)
