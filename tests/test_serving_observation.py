import pytest

from ai_journey.serving_observation import (
    RequestObservation,
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


def test_request_timing_derives_queue_service_and_first_token_latency():
    timing = RequestTiming(10, 20, 25, 40)

    assert timing.queue_time_ns == 10
    assert timing.service_time_ns == 20
    assert timing.time_to_first_token_ns == 15


def test_request_without_a_token_has_no_first_token_latency():
    assert RequestTiming(10, 20, None, 40).time_to_first_token_ns is None


def _observation(request_id="request-1", outcome=ServingOutcome.SUCCESS):
    return RequestObservation(
        request_id=request_id,
        outcome=outcome,
        timing=RequestTiming(10, 20, 25, 40),
        prompt_tokens=128,
        generated_tokens=4 if outcome is ServingOutcome.SUCCESS else 0,
        cache_hit=False,
        retry_count=0,
    )


def test_request_observation_records_terminal_counters():
    observation = _observation()

    assert observation.generated_tokens == 4
    assert observation.retry_count == 0


def test_successful_observation_requires_generated_output():
    values = vars(_observation()) | {"generated_tokens": 0}

    with pytest.raises(ServingObservationError, match="generate"):
        RequestObservation(**values)
