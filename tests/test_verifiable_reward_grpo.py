import math

import pytest

from ai_journey.verifiable_reward_contract import VerifiableRewardError
from ai_journey.verifiable_reward_grpo import group_relative_advantages


def test_group_relative_advantages_are_centered_and_ordered():
    values = group_relative_advantages([0.0, 1.0, 1.0, 0.0])
    assert math.fsum(values) == pytest.approx(0.0, abs=1e-12)
    assert values[1] > values[0]


def test_zero_variance_group_has_zero_advantage_without_nan():
    assert group_relative_advantages([1.0, 1.0]) == (0.0, 0.0)


def test_advantages_reject_nonfinite_or_singleton_groups():
    with pytest.raises(VerifiableRewardError):
        group_relative_advantages([1.0])
    with pytest.raises(VerifiableRewardError):
        group_relative_advantages([1.0, float("nan")])
