from decimal import Decimal

import pytest

from ai_journey.scaling_laws import (
    ComputeOptimalTarget,
    ScalingLawError,
)


def test_chinchilla_convention_sizes_a_100m_parameter_run():
    target = ComputeOptimalTarget(parameter_count=100_000_000)

    assert target.token_target == 2_000_000_000
    assert target.training_flops == 1_200_000_000_000_000_000
    assert target.actual_tokens_per_parameter == Decimal("20")


def test_fractional_ratios_round_tokens_and_compute_upward():
    target = ComputeOptimalTarget(
        parameter_count=3,
        tokens_per_parameter="1.1",
        training_flops_per_parameter_token="2.5",
    )

    assert target.token_target == 4
    assert target.training_flops == 30
    assert target.actual_tokens_per_parameter == Decimal(4) / Decimal(3)


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("parameter_count", True, TypeError),
        ("parameter_count", 0, ScalingLawError),
        ("tokens_per_parameter", float("inf"), ScalingLawError),
        ("tokens_per_parameter", "0", ScalingLawError),
        ("training_flops_per_parameter_token", "nan", ScalingLawError),
    ],
)
def test_scaling_targets_reject_ambiguous_or_nonpositive_controls(
    field, value, exception
):
    values = {"parameter_count": 10}
    values[field] = value

    with pytest.raises(exception):
        ComputeOptimalTarget(**values)
