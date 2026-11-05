import pytest

from ai_journey.scaling_experiment_identity import ExperimentInputs, run_identity
from ai_journey.scaling_experiment_observation import (
    ScalingObservation,
    validate_observation_set,
    verify_observation_identity,
)
from ai_journey.scaling_experiment_protocol import (
    ExperimentPoint,
    ScalingExperimentProtocol,
    build_experiment_grid,
)
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


def inputs():
    return ExperimentInputs(
        source_url="https://example.org/public.txt",
        source_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        code_revision="c" * 40,
    )


def test_observation_identity_recomputes_the_predeclared_run_digest():
    point = ExperimentPoint(100, 2_000, 7)
    result = observation(point=point, run_id=run_identity(point, inputs()))

    assert verify_observation_identity(result, inputs()) is result


def test_observation_identity_rejects_relabelled_measurements():
    result = observation()

    with pytest.raises(ScalingLawError, match="does not match"):
        verify_observation_identity(result, inputs())


def protocol():
    return ScalingExperimentProtocol(
        name="sweep",
        parameter_counts=(10, 20),
        train_token_counts=(100, 200),
        seeds=(1, 2),
    )


def completed_observations():
    return tuple(
        observation(
            point=point,
            run_id=run_identity(point, inputs()),
            trained_tokens=point.train_tokens,
        )
        for point in build_experiment_grid(protocol())
    )


def test_observation_set_requires_exactly_one_run_per_grid_point():
    observations = tuple(reversed(completed_observations()))

    ordered = validate_observation_set(protocol(), inputs(), observations)

    assert tuple(item.point for item in ordered) == build_experiment_grid(protocol())


def test_observation_set_rejects_missing_and_duplicate_points():
    complete = completed_observations()
    with pytest.raises(ScalingLawError, match="complete"):
        validate_observation_set(protocol(), inputs(), complete[:-1])
    with pytest.raises(ScalingLawError, match="duplicate"):
        validate_observation_set(protocol(), inputs(), complete + (complete[0],))
