from decimal import Decimal

import pytest

from ai_journey.scaling_budget import (
    build_isoflop_sweep,
    closest_compute_optimal_candidate,
    compute_optimal_capacity,
    parameter_capacity,
    token_capacity,
)
from ai_journey.scaling_laws import ScalingLawError


def test_budget_inversion_preserves_whole_unit_limits():
    assert token_capacity(compute_budget_flops=1_000, parameter_count=10) == 16
    assert parameter_capacity(compute_budget_flops=1_000, token_count=16) == 10


def test_isoflop_sweep_finds_the_declared_optimal_allocation():
    candidates = build_isoflop_sweep(
        compute_budget_flops=120_000,
        parameter_counts=(20, 10, 40),
    )

    assert [row.parameter_count for row in candidates] == [10, 20, 40]
    assert [row.training_flops for row in candidates] == [120_000] * 3
    assert [row.tokens_per_parameter for row in candidates] == [
        Decimal("200"),
        Decimal("50"),
        Decimal("12.5"),
    ]
    assert closest_compute_optimal_candidate(candidates).parameter_count == 40


def test_isoflop_sweep_rejects_duplicate_sizes_and_unfunded_rows():
    with pytest.raises(ScalingLawError, match="unique"):
        build_isoflop_sweep(
            compute_budget_flops=100,
            parameter_counts=(10, 10),
        )
    with pytest.raises(ScalingLawError, match="cannot fund"):
        build_isoflop_sweep(
            compute_budget_flops=100,
            parameter_counts=(10, 100),
        )


def test_candidate_selector_rejects_empty_or_untyped_inputs():
    with pytest.raises(ScalingLawError, match="nonempty"):
        closest_compute_optimal_candidate(())
    with pytest.raises(TypeError, match="IsoFLOP"):
        closest_compute_optimal_candidate((object(),))


def test_joint_capacity_finds_the_largest_model_at_the_declared_ratio():
    capacity = compute_optimal_capacity(
        compute_budget_flops=120_000,
        tokens_per_parameter="20",
    )

    assert capacity.parameter_count == 31
    assert capacity.token_count == 620
    assert capacity.training_flops == 115_320
    assert capacity.unused_flops == 4_680
    larger = 32 * (32 * 20) * 6
    assert larger > 120_000


def test_joint_capacity_rejects_a_budget_below_one_complete_ratio():
    with pytest.raises(ScalingLawError, match="cannot fund"):
        compute_optimal_capacity(compute_budget_flops=100, tokens_per_parameter="20")
