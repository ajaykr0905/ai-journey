import pytest

from ai_journey.serving_observation import (
    RequestObservation,
    RequestTiming,
    ServingObservationError,
    ServingOutcome,
    canonical_observation_bytes,
    censored_observations,
    require_unique_observation_ids,
    serving_observation_digest,
    validate_observation_for_request,
    validate_observations,
)
from ai_journey.serving_trace import RequestSpec


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


def test_observation_binds_to_trace_identity_and_token_bounds():
    request = RequestSpec("request-1", 0, 128, 8, "a" * 64)

    assert validate_observation_for_request(_observation(), request) == _observation()


def test_observation_rejects_output_beyond_scheduled_maximum():
    request = RequestSpec("request-1", 0, 128, 2, "a" * 64)

    with pytest.raises(ServingObservationError, match="maximum"):
        validate_observation_for_request(_observation(), request)


def test_observation_set_accepts_tied_ordered_receipt_times():
    observations = (_observation("a"), _observation("b"))

    assert validate_observations(observations) == observations


def test_observation_set_rejects_decreasing_receipt_times():
    earlier = RequestObservation(
        **(vars(_observation("b")) | {"timing": RequestTiming(0, 0, 1, 2)})
    )

    with pytest.raises(ServingObservationError, match="ordered"):
        validate_observations((_observation("a"), earlier))


def test_unique_observation_ids_accept_one_terminal_record_per_request():
    observations = (_observation("a"), _observation("b"))

    assert require_unique_observation_ids(observations) == observations


def test_unique_observation_ids_reject_duplicate_terminal_records():
    with pytest.raises(ServingObservationError, match="unique"):
        require_unique_observation_ids((_observation("a"), _observation("a")))


def test_censored_observations_preserve_every_non_success_outcome():
    observations = (
        _observation("a"),
        _observation("b", ServingOutcome.TIMEOUT),
        _observation("c", ServingOutcome.OVERLOADED),
    )

    assert [item.request_id for item in censored_observations(observations)] == [
        "b",
        "c",
    ]


def test_observation_serialization_is_canonical():
    payload = canonical_observation_bytes((_observation("a"), _observation("b")))

    assert payload.startswith(b'[{"cache_hit":false')
    assert b'": ' not in payload


def test_observation_digest_changes_with_terminal_outcome():
    success = (_observation("a"),)
    timeout = (_observation("a", ServingOutcome.TIMEOUT),)

    assert len(serving_observation_digest(success)) == 64
    assert serving_observation_digest(success) != serving_observation_digest(timeout)
