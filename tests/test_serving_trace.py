from fractions import Fraction

import pytest

from ai_journey.serving_trace import (
    RequestSpec,
    ServingTraceError,
    arrival_rate_per_second,
    burst_trace,
    canonical_trace_bytes,
    constant_rate_trace,
    offered_token_rate_per_second,
    peak_arrivals_in_window,
    serving_trace_digest,
    slice_trace_window,
    split_warmup,
    validate_trace,
)

PROMPT_DIGEST = "a" * 64


def test_request_spec_records_schedule_and_prompt_digest_without_raw_content():
    request = RequestSpec("request-1", 25, 128, 32, PROMPT_DIGEST)

    assert request.arrival_ms == 25
    assert request.prompt_sha256 == PROMPT_DIGEST
    assert not hasattr(request, "prompt")


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", ""),
        ("arrival_ms", -1),
        ("prompt_tokens", 0),
        ("max_output_tokens", True),
        ("prompt_sha256", "A" * 64),
    ],
)
def test_request_spec_rejects_ambiguous_or_invalid_fields(field, value):
    values = {
        "request_id": "request-1",
        "arrival_ms": 0,
        "prompt_tokens": 128,
        "max_output_tokens": 32,
        "prompt_sha256": PROMPT_DIGEST,
    }
    values[field] = value

    with pytest.raises((ServingTraceError, TypeError)):
        RequestSpec(**values)


def _request(request_id, arrival_ms):
    return RequestSpec(request_id, arrival_ms, 128, 32, PROMPT_DIGEST)


def test_trace_validation_accepts_tied_arrivals_with_unique_ids():
    trace = (_request("a", 0), _request("b", 0), _request("c", 10))

    assert validate_trace(trace) == trace


@pytest.mark.parametrize(
    "trace,match",
    [
        ((_request("a", 1), _request("b", 0)), "ordered"),
        ((_request("a", 0), _request("a", 1)), "unique"),
    ],
)
def test_trace_validation_rejects_order_and_identity_ambiguity(trace, match):
    with pytest.raises(ServingTraceError, match=match):
        validate_trace(trace)


def test_constant_rate_trace_has_exact_intervals_and_stable_ids():
    trace = constant_rate_trace(
        request_count=3,
        interval_ms=25,
        prompt_tokens=128,
        max_output_tokens=32,
        prompt_sha256=PROMPT_DIGEST,
    )

    assert [request.request_id for request in trace] == [
        "request-000001",
        "request-000002",
        "request-000003",
    ]
    assert [request.arrival_ms for request in trace] == [0, 25, 50]


def test_burst_trace_preserves_tied_arrivals_and_burst_spacing():
    trace = burst_trace(
        burst_count=2,
        requests_per_burst=3,
        inter_burst_ms=100,
        prompt_tokens=16,
        max_output_tokens=8,
        prompt_sha256=PROMPT_DIGEST,
    )

    assert [request.arrival_ms for request in trace] == [0, 0, 0, 100, 100, 100]
    assert len({request.request_id for request in trace}) == 6


def test_arrival_rate_uses_exact_fractional_arithmetic():
    trace = tuple(
        _request(str(index), arrival) for index, arrival in enumerate((0, 30, 60))
    )

    assert arrival_rate_per_second(trace) == Fraction(100, 3)


def test_arrival_rate_rejects_a_zero_duration_burst():
    with pytest.raises(ServingTraceError, match="positive trace span"):
        arrival_rate_per_second((_request("a", 0), _request("b", 0)))


def test_offered_token_rate_includes_prompt_and_requested_output():
    trace = (_request("a", 0), _request("b", 500))

    assert offered_token_rate_per_second(trace, 1000) == Fraction(320, 1)


def test_offered_token_rate_requires_an_observation_tail():
    with pytest.raises(ServingTraceError, match="beyond"):
        offered_token_rate_per_second((_request("a", 10),), 10)


def test_warmup_split_preserves_order_and_all_requests():
    trace = tuple(_request(str(index), index * 10) for index in range(4))

    warmup, measured = split_warmup(trace, 2)

    assert warmup + measured == trace
    assert [request.request_id for request in measured] == ["2", "3"]


def test_warmup_split_cannot_consume_the_measured_sample():
    with pytest.raises(ServingTraceError, match="leave"):
        split_warmup((_request("a", 0),), 1)


def test_trace_window_uses_half_open_absolute_boundaries():
    trace = tuple(
        _request(str(index), arrival) for index, arrival in enumerate((0, 10, 20))
    )

    selected = slice_trace_window(trace, start_ms=10, end_ms=20)

    assert [request.arrival_ms for request in selected] == [10]


def test_trace_window_rejects_empty_selection():
    with pytest.raises(ServingTraceError, match="no requests"):
        slice_trace_window((_request("a", 0),), start_ms=10, end_ms=20)


def test_peak_arrivals_counts_tied_bursts_inside_half_open_windows():
    trace = tuple(
        _request(str(index), arrival) for index, arrival in enumerate((0, 0, 9, 10, 19))
    )

    assert peak_arrivals_in_window(trace, 10) == 3


def test_trace_digest_binds_order_schedule_and_token_demand():
    trace = (_request("a", 0), _request("b", 10))
    changed = (_request("a", 0), RequestSpec("b", 11, 128, 32, PROMPT_DIGEST))

    assert canonical_trace_bytes(trace).startswith(b'[{"arrival_ms":0')
    assert len(serving_trace_digest(trace)) == 64
    assert serving_trace_digest(trace) != serving_trace_digest(changed)
