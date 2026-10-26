from contextlib import contextmanager
import http.client
import threading

import pytest

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_http import make_server
from ai_journey.ddp_service import GenerationRequest, GenerationService, ServiceDraining
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def service():
    return GenerationService(
        GPT2LanguageModel(
            GPT2Config(
                vocab_size=8,
                context_length=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                embedding_dropout=0,
                attention_dropout=0,
                residual_dropout=0,
            )
        )
    )


def request(name="fresh"):
    return GenerationRequest(
        name, (1,), SamplingConfig(max_new_tokens=2, temperature=0)
    )


def test_metrics_count_cache_replay_without_new_generation():
    instance = service()
    first = instance.generate(request())
    assert instance.generate(request()) == first
    values = instance.metrics.snapshot()
    assert values["requests"]["success"] == 2
    assert values["generated_tokens"] == 2
    assert values["cache_hits"] == 1
    assert values["duration_seconds"]["count"] == 2


def test_metrics_retain_cancellation_deadline_rejection_and_partial_failure():
    instance = service()
    instance.generate(request("cancel"), cancelled=lambda: True)
    instance.generate(request("deadline"), deadline=0)
    with pytest.raises(ValueError):
        instance.generate(GenerationRequest("invalid", (99,), SamplingConfig()))

    def fail(index, token):
        raise RuntimeError("client callback failed")

    with pytest.raises(RuntimeError):
        instance.generate(request("failure"), on_token=fail)
    instance.drain(timeout_seconds=0)
    with pytest.raises(ServiceDraining):
        instance.generate(request("draining"))
    values = instance.metrics.snapshot()
    assert values["requests"] == {
        "success": 0,
        "cancelled": 1,
        "deadline": 1,
        "error": 1,
        "rejected": 2,
    }
    assert values["generated_tokens"] == 1


@contextmanager
def server_for(instance):
    server = make_server(instance)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_real_http_metrics_scrape_remains_available_during_drain():
    instance = service()
    instance.generate(request())
    instance.drain(timeout_seconds=0)
    with server_for(instance) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        client.request("GET", "/metrics")
        response = client.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith(
            "text/plain; version=0.0.4"
        )
        body = response.read().decode()
        assert 'ai_journey_serving_requests_total{outcome="success"} 1' in body
        assert "ai_journey_serving_generated_tokens_total 2" in body
        assert "fresh" not in body
        client.close()
