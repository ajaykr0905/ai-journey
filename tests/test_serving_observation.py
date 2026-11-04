import pytest

from ai_journey.serving_observation import ServingOutcome


def test_serving_outcomes_have_a_fixed_low_cardinality_vocabulary():
    assert {outcome.value for outcome in ServingOutcome} == {
        "success",
        "timeout",
        "overloaded",
        "cancelled",
        "error",
    }


def test_serving_outcome_rejects_unbounded_user_supplied_labels():
    with pytest.raises(ValueError):
        ServingOutcome("customer-specific-error")
