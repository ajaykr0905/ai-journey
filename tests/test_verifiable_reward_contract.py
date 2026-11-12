import pytest

from ai_journey.verifiable_reward_contract import (
    RewardRunConfig,
    VerifiableRewardError,
)


def valid_config():
    return {
        "seed": 69,
        "candidates_per_task": 4,
        "max_tasks": 20,
        "max_response_bytes": 4096,
        "max_attempts": 2,
    }


def test_config_accepts_bounded_controls():
    config = RewardRunConfig.from_mapping(valid_config())

    assert config == RewardRunConfig(69, 4, 20, 4096, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("candidates_per_task", 1),
        ("max_tasks", 0),
        ("max_response_bytes", 15),
        ("max_attempts", 9),
        ("seed", True),
    ],
)
def test_config_rejects_invalid_bounds_and_types(field, value):
    payload = valid_config()
    payload[field] = value

    with pytest.raises(VerifiableRewardError, match=field):
        RewardRunConfig.from_mapping(payload)


def test_config_rejects_unknown_and_missing_fields():
    payload = valid_config()
    payload["unbounded"] = True

    with pytest.raises(VerifiableRewardError, match="fields mismatch"):
        RewardRunConfig.from_mapping(payload)
