import pytest

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_service import GenerationRequest, GenerationService, TokenBucket
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def service_and_request():
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
    request = GenerationRequest(
        "retained", (1, 2), SamplingConfig(max_new_tokens=3, temperature=0)
    )
    return service, request


def test_cached_response_obeys_cancellation_without_discarding_retained_result():
    service, request = service_and_request()
    complete = service.generate(request)
    emitted = []
    cancelled = service.generate(
        request,
        cancelled=lambda: True,
        on_token=lambda index, token: emitted.append((index, token)),
    )
    assert cancelled["finish_reason"] == "cancelled"
    assert cancelled["tokens"] == []
    assert emitted == []
    assert service.generate(request) == complete
    assert service.completed_requests == 1


def test_cached_stream_stops_between_tokens_when_cancelled():
    service, request = service_and_request()
    complete = service.generate(request)
    emitted = []

    def emit(index, token):
        emitted.append((index, token))

    partial = service.generate(request, cancelled=lambda: bool(emitted), on_token=emit)
    assert partial["finish_reason"] == "cancelled"
    assert partial["tokens"] == complete["tokens"][:1]
    assert emitted == list(enumerate(complete["tokens"][:1]))
    assert service.generate(request) == complete


def test_cached_response_obeys_expired_deadline(monkeypatch):
    service, request = service_and_request()
    complete = service.generate(request)
    monkeypatch.setattr("ai_journey.ddp_service.time.monotonic", lambda: 10.0)
    expired = service.generate(request, deadline=10.0)
    assert expired["finish_reason"] == "deadline"
    assert expired["tokens"] == []
    assert service.generate(request) == complete


@pytest.mark.parametrize(
    "controls", [{"cancelled": True}, {"on_token": 1}, {"deadline": float("nan")}]
)
def test_cached_responses_validate_controls_like_uncached_requests(controls):
    service, request = service_and_request()
    service.generate(request)
    with pytest.raises((TypeError, ValueError)):
        service.generate(request, **controls)


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), -float("inf"), True])
def test_admission_rejects_invalid_initial_clock_reading(reading):
    with pytest.raises(ValueError, match="clock"):
        TokenBucket(5, 1, clock=lambda: reading)


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_admission_clock_cannot_refill_or_mutate_budget(reading):
    readings = [0.0]
    bucket = TokenBucket(5, 1, clock=lambda: readings[0])
    bucket.consume(5)
    readings[0] = reading
    with pytest.raises(ValueError, match="clock"):
        bucket.consume(1)
    assert bucket.tokens == 0
    assert bucket.updated == 0
    readings[0] = 1.0
    bucket.consume(1)
    assert bucket.tokens == 0


def test_out_of_vocabulary_prompt_is_rejected_before_charging_admission():
    bucket = TokenBucket(5, 1, clock=lambda: 0.0)
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), admission=bucket
    )
    invalid = GenerationRequest(
        "invalid", (8,), SamplingConfig(max_new_tokens=3, temperature=0)
    )
    with pytest.raises(ValueError, match="vocabulary"):
        service.generate(invalid)
    assert bucket.tokens == 5
    assert service.completed_requests == 0
    valid = GenerationRequest(
        "valid", (1, 2), SamplingConfig(max_new_tokens=3, temperature=0)
    )
    assert len(service.generate(valid)["tokens"]) == 3
