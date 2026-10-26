import threading

import pytest

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_service import GenerationRequest, GenerationService
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def test_drain_marks_unready_rejects_new_work_and_waits_for_admitted_request(
    monkeypatch,
):
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
    request = GenerationRequest("active", (1,), SamplingConfig(max_new_tokens=1))
    entered = threading.Event()
    release = threading.Event()
    results = []

    def blocked_generation(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {"tokens": [1], "prompt_tokens": 1, "finish_reason": "length"}

    monkeypatch.setattr("ai_journey.ddp_service.generate_tokens", blocked_generation)
    worker = threading.Thread(target=lambda: results.append(service.generate(request)))
    worker.start()
    try:
        assert entered.wait(timeout=2)
        assert not service.drain(timeout_seconds=0)
        health = service.health()
        assert not health["ready"]
        assert health["active_requests"] == 1
        with pytest.raises(RuntimeError, match="draining"):
            service.generate(request)
    finally:
        release.set()
        worker.join(timeout=5)
    assert service.drain(timeout_seconds=1)
    assert service.health()["active_requests"] == 0
    assert service.health()["completed_requests"] == 1
    assert results[0]["tokens"] == [1]


def test_failed_request_always_releases_drain_accounting(monkeypatch):
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))

    def fail(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("ai_journey.ddp_service.generate_tokens", fail)
    request = GenerationRequest("failed", (1,), SamplingConfig(max_new_tokens=1))
    with pytest.raises(RuntimeError, match="model failed"):
        service.generate(request)
    assert service.drain(timeout_seconds=0)
    assert service.health()["active_requests"] == 0


@pytest.mark.parametrize("timeout", [True, -1, float("nan"), float("inf"), 61])
def test_drain_rejects_invalid_deadlines_without_changing_readiness(timeout):
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
    with pytest.raises(ValueError, match="timeout"):
        service.drain(timeout_seconds=timeout)
    assert service.health()["ready"]
