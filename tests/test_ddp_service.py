import pytest
from ai_journey.ddp_service import GenerationRequest
from ai_journey.ddp_service import GenerationService
from ai_journey.ddp_generation import SamplingConfig
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_config import GPT2Config
import torch


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


def test_service_isolated_snapshot_survives_caller_model_mutation():
    model = GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1))
    service = GenerationService(model)
    expected = service.generate(request())
    with torch.no_grad():
        model.wte.weight.zero_()
    assert service.generate(request()) == expected
    assert service.health()["completed_requests"] == 2
    assert service.health()["generated_tokens"] == 6
