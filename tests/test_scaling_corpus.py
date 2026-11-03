from decimal import Decimal

import pytest

from ai_journey.scaling_corpus import plan_corpus_reuse
from ai_journey.scaling_laws import ScalingLawError


def test_corpus_plan_exposes_reuse_for_the_100m_target():
    plan = plan_corpus_reuse(
        target_tokens=2_000_000_000,
        unique_train_tokens=69_626,
    )

    assert plan.full_epochs == 28_724
    assert plan.final_epoch_tokens == 62_776
    assert plan.total_epochs == Decimal(2_000_000_000) / Decimal(69_626)
    assert plan.repeats_data is True
    assert plan.distinct_token_fraction == Decimal(69_626) / Decimal(2_000_000_000)


def test_corpus_plan_handles_one_partial_pass_without_claiming_reuse():
    plan = plan_corpus_reuse(target_tokens=80, unique_train_tokens=100)

    assert plan.full_epochs == 0
    assert plan.final_epoch_tokens == 80
    assert plan.total_epochs == Decimal("0.8")
    assert plan.repeats_data is False
    assert plan.distinct_token_fraction == 1


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("target_tokens", 0, ScalingLawError),
        ("target_tokens", True, TypeError),
        ("unique_train_tokens", -1, ScalingLawError),
    ],
)
def test_corpus_plan_rejects_invalid_counts(field, value, exception):
    values = {"target_tokens": 10, "unique_train_tokens": 10}
    values[field] = value

    with pytest.raises(exception):
        plan_corpus_reuse(**values)
