from decimal import Decimal

import pytest

from ai_journey.scaling_laws import ScalingLawError
from ai_journey.scaling_runtime import (
    RuntimeAssumptions,
    estimate_runtime_and_cost,
)


def test_runtime_estimate_accounts_for_parallel_wall_time_and_total_device_hours():
    assumptions = RuntimeAssumptions(
        sustained_flops_per_second="100000000000000",
        accelerator_count=2,
        hourly_cost_per_accelerator="3.50",
    )

    estimate = estimate_runtime_and_cost(1_200_000_000_000_000_000, assumptions)

    assert estimate.seconds == Decimal("6000")
    assert estimate.accelerator_hours == Decimal(10) / Decimal(3)
    assert estimate.monetary_cost == Decimal(35) / Decimal(3)


def test_runtime_estimate_does_not_round_small_runs_to_zero():
    estimate = estimate_runtime_and_cost(
        1,
        RuntimeAssumptions(
            sustained_flops_per_second="3",
            accelerator_count=1,
            hourly_cost_per_accelerator="1",
        ),
    )

    assert estimate.seconds == Decimal(1) / Decimal(3)
    assert estimate.monetary_cost > 0


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("sustained_flops_per_second", "0", ScalingLawError),
        ("accelerator_count", True, TypeError),
        ("hourly_cost_per_accelerator", "nan", ScalingLawError),
    ],
)
def test_runtime_assumptions_reject_invalid_controls(field, value, exception):
    values = {
        "sustained_flops_per_second": "100",
        "accelerator_count": 1,
        "hourly_cost_per_accelerator": "1",
    }
    values[field] = value

    with pytest.raises(exception):
        RuntimeAssumptions(**values)
