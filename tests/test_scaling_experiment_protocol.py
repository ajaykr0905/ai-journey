import pytest

from ai_journey.scaling_experiment_protocol import ExperimentPoint
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
