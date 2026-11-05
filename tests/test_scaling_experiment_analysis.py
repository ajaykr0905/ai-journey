import pytest

from ai_journey.scaling_experiment_analysis import aggregate_replicates
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
