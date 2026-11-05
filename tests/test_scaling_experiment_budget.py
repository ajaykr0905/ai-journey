import pytest

from ai_journey.scaling_experiment_budget import (
    estimate_experiment_budget,
    point_training_flops,
)
from ai_journey.scaling_experiment_protocol import (
    ExperimentPoint,
    ScalingExperimentProtocol,
)


def test_point_compute_uses_declared_dense_training_convention():
    point = ExperimentPoint(parameter_count=10, train_tokens=100, seed=3)

    assert point_training_flops(point) == 6_000
    assert point_training_flops(point, flops_per_parameter_token=8) == 8_000


def test_experiment_budget_counts_every_seed_and_coordinate():
    protocol = ScalingExperimentProtocol(
        name="budgeted",
        parameter_counts=(10, 20),
        train_token_counts=(100, 200),
        seeds=(1, 2),
    )

    budget = estimate_experiment_budget(protocol)

    assert budget.run_count == 8
    assert budget.total_training_flops == 108_000
    assert budget.maximum_run_flops == 24_000


def test_compute_accounting_rejects_invalid_conventions():
    point = ExperimentPoint(parameter_count=10, train_tokens=100, seed=3)

    with pytest.raises((TypeError, ValueError)):
        point_training_flops(point, flops_per_parameter_token=0)
