import pytest

from ai_journey.verifiable_reward_budget import validate_run_budget
from ai_journey.verifiable_reward_contract import RewardRunConfig, VerifiableRewardError


CONFIG = RewardRunConfig(1, 4, 3, 100, 2)


def test_budget_returns_hard_upper_bound_on_attempts():
    assert validate_run_budget(task_count=3, group_count=3, config=CONFIG) == 24


def test_budget_rejects_missing_groups_and_task_overflow():
    with pytest.raises(VerifiableRewardError, match="exactly one"):
        validate_run_budget(task_count=2, group_count=1, config=CONFIG)
    with pytest.raises(VerifiableRewardError, match="task count"):
        validate_run_budget(task_count=4, group_count=4, config=CONFIG)
