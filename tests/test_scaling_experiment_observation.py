import pytest

from ai_journey.scaling_experiment_observation import ScalingObservation
from ai_journey.scaling_experiment_protocol import ExperimentPoint
from ai_journey.scaling_laws import ScalingLawError


def observation(**overrides):
    point = overrides.pop("point", ExperimentPoint(100, 2_000, 7))
    values = {
        "run_id": "a" * 64,
        "point": point,
        "validation_loss": 2.5,
        "trained_tokens": point.train_tokens,
        "optimizer_steps": 10,
        "elapsed_seconds": 1.25,
    }
    values.update(overrides)
    return ScalingObservation(**values)


def test_observation_records_a_completed_predeclared_run():
    result = observation()

    assert result.validation_loss == 2.5
    assert result.trained_tokens == 2_000
    assert result.optimizer_steps == 10
    assert result.elapsed_seconds == 1.25


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("run_id", "A" * 64, "run_id"),
        ("validation_loss", float("nan"), "validation_loss"),
        ("trained_tokens", 1_999, "predeclared"),
        ("optimizer_steps", 0, "positive"),
        ("elapsed_seconds", float("inf"), "elapsed_seconds"),
    ],
)
def test_observation_rejects_invalid_or_partial_measurements(field, value, message):
    with pytest.raises((ScalingLawError, TypeError), match=message):
        observation(**{field: value})
