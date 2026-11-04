import pytest

from ai_journey.serving_trace import (
    RequestSpec,
    ServingTraceError,
    constant_rate_trace,
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
