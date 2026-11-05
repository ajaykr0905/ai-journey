import pytest

from ai_journey.scaling_experiment_protocol import (
    ExperimentPoint,
    ScalingExperimentProtocol,
    build_experiment_grid,
)
from ai_journey.scaling_laws import ScalingLawError


def test_experiment_point_records_one_predeclared_coordinate():
    point = ExperimentPoint(parameter_count=1_000_000, train_tokens=20_000_000, seed=7)

    assert point.parameter_count == 1_000_000
    assert point.train_tokens == 20_000_000
    assert point.seed == 7


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("parameter_count", 0, ScalingLawError),
        ("train_tokens", True, TypeError),
        ("seed", -1, ScalingLawError),
        ("seed", 1.5, TypeError),
    ],
)
def test_experiment_point_rejects_invalid_coordinates(field, value, error):
    values = {"parameter_count": 1, "train_tokens": 1, "seed": 0}
    values[field] = value

    with pytest.raises(error):
        ExperimentPoint(**values)


def test_protocol_requires_ordered_model_token_and_seed_axes():
    protocol = ScalingExperimentProtocol(
        name="day-62-bounded-sweep",
        parameter_counts=(100_000, 200_000, 400_000),
        train_token_counts=(1_000_000, 2_000_000),
        seeds=(3, 7),
    )

    assert protocol.parameter_counts == (100_000, 200_000, 400_000)
    assert protocol.train_token_counts == (1_000_000, 2_000_000)
    assert protocol.seeds == (3, 7)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("name", "", "name"),
        ("parameter_counts", (2, 1), "strictly increasing"),
        ("train_token_counts", (1,), "at least two"),
        ("seeds", (0, 0), "strictly increasing"),
    ],
)
def test_protocol_rejects_ambiguous_axes(field, value, message):
    values = {
        "name": "sweep",
        "parameter_counts": (1, 2),
        "train_token_counts": (2, 4),
        "seeds": (0, 1),
    }
    values[field] = value

    with pytest.raises(ScalingLawError, match=message):
        ScalingExperimentProtocol(**values)


def test_grid_expands_every_declared_coordinate_in_stable_order():
    protocol = ScalingExperimentProtocol(
        name="sweep",
        parameter_counts=(10, 20),
        train_token_counts=(100, 200),
        seeds=(1, 2),
    )

    grid = build_experiment_grid(protocol)

    assert len(grid) == 8
    assert grid[:3] == (
        ExperimentPoint(10, 100, 1),
        ExperimentPoint(10, 100, 2),
        ExperimentPoint(10, 200, 1),
    )
    assert grid[-1] == ExperimentPoint(20, 200, 2)
    assert len(set(grid)) == len(grid)


def test_grid_rejects_non_protocol_inputs():
    with pytest.raises(TypeError, match="ScalingExperimentProtocol"):
        build_experiment_grid(object())
