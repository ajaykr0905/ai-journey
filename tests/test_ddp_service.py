import pytest
from ai_journey.ddp_service import GenerationRequest
from ai_journey.ddp_service import GenerationService
from ai_journey.ddp_service import TokenBucket, AdmissionRejected
from ai_journey.ddp_service import event_frame
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
    assert service.health()["completed_requests"] == 1
    assert service.health()["generated_tokens"] == 3


def test_idempotent_retries_are_isolated_and_conflicting_ids_fail():
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), cache_capacity=1
    )
    first = service.generate(request())
    first["tokens"].clear()
    assert len(service.generate(request())["tokens"]) == 3
    conflicting = GenerationRequest("request-1", (3, 4), request().sampling)
    with pytest.raises(ValueError, match="already used"):
        service.generate(conflicting)
    service.generate(request("another"))
    service.generate(conflicting)
    assert service.health()["completed_requests"] == 3


def test_admission_budgets_refill_and_idempotent_retries_cost_nothing():
    clock = [0.0]
    bucket = TokenBucket(5, 1, clock=lambda: clock[0])
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), admission=bucket
    )
    service.generate(request())
    service.generate(request())
    with pytest.raises(AdmissionRejected) as error:
        service.generate(request("second"))
    assert error.value.retry_after == 5
    clock[0] = 5
    service.generate(request("second"))
    assert service.health()["completed_requests"] == 2


def test_token_stream_frames_match_response_and_replay_on_retry():
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
    emitted = []
    result = service.generate(
        request(), on_token=lambda index, token: emitted.append((index, token))
    )
    assert emitted == list(enumerate(result["tokens"]))
    replayed = []
    service.generate(
        request(), on_token=lambda index, token: replayed.append((index, token))
    )
    assert replayed == emitted
    assert (
        event_frame("done", {"text": "\nattack"})
        == b'event: done\ndata: {"text":"\\nattack"}\n\n'
    )
