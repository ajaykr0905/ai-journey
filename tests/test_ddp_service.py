import pytest
from ai_journey.ddp_service import GenerationRequest
from ai_journey.ddp_generation import SamplingConfig


def request(identifier="request-1"):
    return GenerationRequest(
        identifier, (1, 2), SamplingConfig(max_new_tokens=3, temperature=0)
    )


def test_request_parser_binds_prompt_and_sampling_controls():
    parsed = GenerationRequest.from_mapping(
        {
            "request_id": "request-1",
            "prompt": [1, 2],
            "sampling": {"max_new_tokens": 3, "temperature": 0},
        }
    )
    assert parsed == request()
    assert parsed.fingerprint() == request().fingerprint()
    assert parsed.fingerprint() != request("request-2").fingerprint()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"request_id": "../escape", "prompt": [1], "sampling": {}},
        {"request_id": "x", "prompt": [True], "sampling": {}},
        {"request_id": "x", "prompt": [1], "sampling": {"max_new_tokens": 129}},
    ],
)
def test_invalid_service_requests_fail_before_model_execution(payload):
    with pytest.raises((ValueError, TypeError)):
        GenerationRequest.from_mapping(payload)
