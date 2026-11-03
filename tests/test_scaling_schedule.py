import pytest

from ai_journey.scaling_laws import ScalingLawError
from ai_journey.scaling_schedule import (
    TrainingBatchGeometry,
    align_token_target,
    validate_overshoot,
)


def test_schedule_aligns_targets_to_complete_distributed_updates():
    geometry = TrainingBatchGeometry(
        sequence_length=2_048,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
        data_parallel_size=8,
    )

    schedule = align_token_target(2_000_000_000, geometry)

    assert geometry.tokens_per_optimizer_step == 131_072
    assert schedule.optimizer_steps == 15_259
    assert schedule.scheduled_tokens == 2_000_027_648
    assert schedule.overshoot_tokens == 27_648
    assert schedule.exact is False


def test_schedule_marks_exact_targets_without_overshoot():
    geometry = TrainingBatchGeometry(10, 2, 3, 4)
    schedule = align_token_target(240, geometry)

    assert schedule.optimizer_steps == 1
    assert schedule.exact is True
    assert validate_overshoot(schedule, maximum_fraction=0) is schedule


def test_overshoot_policy_rejects_material_batch_rounding():
    schedule = align_token_target(10, TrainingBatchGeometry(8, 1, 1, 1))

    with pytest.raises(ScalingLawError, match="overshoot"):
        validate_overshoot(schedule, maximum_fraction=0.5)
    assert validate_overshoot(schedule, maximum_fraction=0.6) is schedule


@pytest.mark.parametrize("value", [-0.1, 1, float("nan"), True])
def test_overshoot_policy_rejects_invalid_fractions(value):
    schedule = align_token_target(8, TrainingBatchGeometry(8, 1, 1, 1))

    with pytest.raises(ScalingLawError, match="maximum_fraction"):
        validate_overshoot(schedule, maximum_fraction=value)
