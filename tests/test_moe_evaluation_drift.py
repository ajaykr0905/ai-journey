import copy
import math

import pytest

from ai_journey.moe_evaluation_drift import compare_moe_route_distributions


def test_uniform_routes_and_exposure_scaling_have_zero_drift():
    result = compare_moe_route_distributions([2, 2, 2, 2], [50, 50, 50, 50])
    assert result["reference"]["assignments"] == 8
    assert result["observed"]["assignments"] == 200
    assert result["jensen_shannon_nats"] == 0.0
    assert result["total_variation"] == 0.0
    assert result["reference"]["normalized_entropy"] == 1.0
    assert result["reference"]["effective_experts"] == 4.0


def test_disjoint_collapsed_routes_reach_known_divergence_limits():
    result = compare_moe_route_distributions([10, 0, 0, 0], [0, 0, 0, 20])
    assert result["jensen_shannon_nats"] == math.log(2)
    assert result["total_variation"] == 1.0
    assert result["observed"]["normalized_entropy"] == 0.0
    assert result["observed"]["maximum_fraction"] == 1.0
    assert result["observed"]["effective_experts"] == 1.0


@pytest.mark.parametrize(
    "reference,observed", [([0, 0], [0, 0]), ([0, 0], [1, 2]), ([1, 2], [0, 0])]
)
def test_absent_exposure_is_null_not_a_fabricated_distribution(reference, observed):
    result = compare_moe_route_distributions(reference, observed)
    assert result["jensen_shannon_nats"] is None
    assert result["total_variation"] is None
    empty = result["reference"] if sum(reference) == 0 else result["observed"]
    assert empty["fractions"] is empty["normalized_entropy"] is None


def test_drift_is_symmetric_and_preserves_inputs():
    left, right = [3, 1, 0, 6], [2, 4, 3, 1]
    originals = copy.deepcopy((left, right))
    result = compare_moe_route_distributions(left, right)
    swapped = compare_moe_route_distributions(right, left)
    permuted = compare_moe_route_distributions(left[::-1], right[::-1])
    for field in ("jensen_shannon_nats", "total_variation"):
        assert result[field] == swapped[field] == permuted[field]
    assert 0 <= result["jensen_shannon_nats"] <= math.log(2)
    assert (left, right) == originals
    assert "not evidence of specialization" in result["scope"]


@pytest.mark.parametrize(
    "bad", [[True, 0], [1.0, 0], [-1, 2], [2**31, 0], [1], [0] * 65, "10", None]
)
def test_histogram_requires_bounded_real_integer_counts(bad):
    with pytest.raises(ValueError):
        compare_moe_route_distributions(bad, [1, 2])


def test_mismatched_expert_spaces_are_rejected():
    with pytest.raises(ValueError, match="identities"):
        compare_moe_route_distributions([1, 2], [1, 2, 3])
