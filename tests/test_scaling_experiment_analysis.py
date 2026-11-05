import pytest

from ai_journey.scaling_experiment_analysis import (
    AggregatedPoint,
    aggregate_replicates,
    fit_compute_loss,
    largest_model_holdout,
)
from ai_journey.scaling_experiment_observation import ScalingObservation
from ai_journey.scaling_experiment_protocol import ExperimentPoint
from ai_journey.scaling_laws import ScalingLawError


def observation(parameters, tokens, seed, loss):
    point = ExperimentPoint(parameters, tokens, seed)
    return ScalingObservation(
        run_id=f"{seed + 1:064x}",
        point=point,
        validation_loss=loss,
        trained_tokens=tokens,
        optimizer_steps=10,
        elapsed_seconds=1.0,
    )


def test_replicate_aggregation_preserves_each_model_token_coordinate():
    rows = (
        observation(10, 100, 0, 4.0),
        observation(10, 100, 1, 2.0),
        observation(20, 100, 0, 3.0),
        observation(20, 100, 1, 1.0),
    )

    result = aggregate_replicates(tuple(reversed(rows)))

    assert [(row.parameter_count, row.train_tokens) for row in result] == [
        (10, 100),
        (20, 100),
    ]
    assert result[0].mean_validation_loss == 3.0
    assert result[0].sample_standard_deviation == pytest.approx(2**0.5)
    assert result[0].replicate_count == 2


def test_replicate_aggregation_rejects_single_seed_coordinates():
    with pytest.raises(ScalingLawError, match="at least two"):
        aggregate_replicates((observation(10, 100, 0, 2.0),))


def aggregate_point(parameters, tokens, loss=2.0):
    return AggregatedPoint(parameters, tokens, loss, 0.1, 2)


def test_holdout_reserves_largest_model_at_each_token_budget():
    points = tuple(
        aggregate_point(parameters, tokens)
        for parameters in (10, 20, 40)
        for tokens in (100, 200)
    )

    split = largest_model_holdout(points)

    assert [(row.parameter_count, row.train_tokens) for row in split.holdout] == [
        (40, 100),
        (40, 200),
    ]
    assert len(split.training) == 4
    assert not set(split.training) & set(split.holdout)


def test_holdout_rejects_splits_too_small_for_power_law_fitting():
    points = (aggregate_point(10, 100), aggregate_point(20, 100))

    with pytest.raises(ScalingLawError, match="three training"):
        largest_model_holdout(points)


def test_compute_loss_fit_uses_exact_analytical_compute_coordinates():
    compute = (600.0, 1_200.0, 2_400.0, 4_800.0)
    points = tuple(
        aggregate_point(parameters, 10, loss=10 * flops**-0.1)
        for parameters, flops in zip((10, 20, 40, 80), compute, strict=True)
    )

    fit = fit_compute_loss(points)

    assert fit.exponent == pytest.approx(-0.1)
    assert fit.coefficient == pytest.approx(10.0)
    assert fit.r_squared_log_space == pytest.approx(1.0)


def test_compute_loss_fit_rejects_insufficient_training_points():
    with pytest.raises(ScalingLawError, match="at least three"):
        fit_compute_loss((aggregate_point(10, 10), aggregate_point(20, 10)))
