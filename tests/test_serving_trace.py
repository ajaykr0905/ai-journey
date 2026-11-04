import pytest

from ai_journey.serving_trace import RequestSpec, ServingTraceError

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
