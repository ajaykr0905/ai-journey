import pytest

from ai_journey.verifiable_reward_checkpoint import RewardCheckpoint
from ai_journey.verifiable_reward_contract import VerifiableRewardError


SHA = "a" * 64


def test_checkpoint_round_trips_group_boundary_and_digests():
    checkpoint = RewardCheckpoint(SHA, 2, SHA)
    assert RewardCheckpoint.from_mapping(checkpoint.to_mapping()) == checkpoint


def test_checkpoint_rejects_malformed_digest_and_boolean_index():
    with pytest.raises(VerifiableRewardError, match="digests"):
        RewardCheckpoint("bad", 0, SHA)
    with pytest.raises(VerifiableRewardError, match="integer"):
        RewardCheckpoint.from_mapping(
            {"task_set_sha256": SHA, "next_group_index": True, "attempts_sha256": SHA}
        )
