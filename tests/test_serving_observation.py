import pytest

from ai_journey.serving_observation import (
    RequestTiming,
    ServingObservationError,
    ServingOutcome,
)


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


def test_request_timing_accepts_a_monotonic_lifecycle():
    timing = RequestTiming(10, 20, 25, 40)

    assert timing.first_token_ns == 25


@pytest.mark.parametrize(
    "values",
    [(20, 10, 15, 30), (10, 20, 19, 30), (10, 20, 31, 30), (-1, 0, None, 1)],
)
def test_request_timing_rejects_nonmonotonic_or_negative_clocks(values):
    with pytest.raises(ServingObservationError):
        RequestTiming(*values)
