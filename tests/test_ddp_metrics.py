import math
import pytest
import torch
from ai_journey.ddp_metrics import ddp_loss_scale, token_nll, weighted_mean
from ai_journey.ddp_metrics import binary_metrics
from ai_journey.ddp_metrics import calibration_metrics


def test_uneven_rank_gradient_scaling_matches_global_token_mean():
    local_means = [2.0, 8.0]
    counts = [3, 1]
    averaged = (
        sum(
            mean * ddp_loss_scale(count, 4, 2)
            for mean, count in zip(local_means, counts)
        )
        / 2
    )
    assert averaged == weighted_mean([(6.0, 3), (8.0, 1)]) == 3.5


def test_per_token_cross_entropy_is_stable_for_large_logits():
    logits = torch.tensor(
        [[[10000.0, 10000.0], [-10000.0, -10000.0]]], requires_grad=True
    )
    losses = token_nll(logits, torch.tensor([[0, 1]]))
    assert losses.tolist()[0] == pytest.approx([math.log(2)] * 2)
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("totals", [[], [(0, 0)], [(float("nan"), 1)], [(1, -1)]])
def test_invalid_weighted_totals_fail(totals):
    with pytest.raises(ValueError):
        weighted_mean(totals)


def test_sensitivity_specificity_keep_distinct_denominators():
    result = binary_metrics([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["specificity"] == 0.5
    assert result["accuracy"] == 0.6
    assert binary_metrics([0], [0])["sensitivity"] is None
    assert binary_metrics([1], [0])["precision"] is None


def test_calibration_has_independent_perfect_and_uniform_oracles():
    perfect = calibration_metrics([[1, 0], [0, 1]], [0, 1])
    assert perfect["brier_score"] == perfect["expected_calibration_error"] == 0
    uniform = calibration_metrics([[0.5, 0.5], [0.5, 0.5]], [0, 1])
    assert uniform["brier_score"] == 0.5
    assert uniform["expected_calibration_error"] == 0
    assert sum(record["count"] for record in uniform["bins"]) == 2
    with pytest.raises(ValueError, match="sum to one"):
        calibration_metrics([[0.1, 0.1]], [0])
