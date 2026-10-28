from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import threading
import time

import pytest

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_service import (
    GenerationRequest,
    GenerationService,
    ServiceDraining,
    TokenBucket,
)
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from test_ddp_metrics_integration import server_for


def _get(server, path):
    connection = http.client.HTTPConnection(*server.server_address, timeout=1)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == 200
        return response.read().decode()
    finally:
        connection.close()


def _samples(text):
    return {
        name: float(value)
        for line in text.splitlines()
        if line and not line.startswith("#")
        for name, value in (line.rsplit(" ", 1),)
    }


def _assert_state(server, *, active, running, queued, draining):
    health = json.loads(_get(server, "/health"))
    metrics = _samples(_get(server, "/metrics"))
    expected = {
        "active_requests": active,
        "running_requests": running,
        "queued_requests": queued,
        "request_capacity": 2,
        "draining": draining,
    }
    assert {name: health[name] for name in expected} == expected
    assert health["ready"] is not draining
    for name, value in expected.items():
        assert metrics[f"ai_journey_serving_{name}"] == value
    return metrics


def test_http_observes_queue_cancellation_and_drain_during_paused_generation(
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr("ai_journey.ddp_service.time.perf_counter", lambda: clock[0])
    bucket = TokenBucket(8, 1, clock=lambda: 0.0)
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)),
        admission=bucket,
        max_inflight_requests=2,
    )
    entered, release, cancelled, queued_entered = (threading.Event() for _ in range(4))

    def queued_cancelled():
        queued_entered.set()
        return cancelled.is_set()

    def pause(index, token):
        if index == 0:
            entered.set()
            assert release.wait(timeout=5)

    def request(identifier):
        return GenerationRequest(
            identifier, (1, 2), SamplingConfig(max_new_tokens=2, temperature=0)
        )

    with server_for(service) as server, ThreadPoolExecutor(max_workers=2) as workers:
        _assert_state(server, active=0, running=0, queued=0, draining=False)
        active = workers.submit(
            service.generate, request("private-active"), on_token=pause
        )
        try:
            assert entered.wait(timeout=3)
            clock[0] = 2.0
            queued = workers.submit(
                service.generate, request("private-queued"), cancelled=queued_cancelled
            )
            assert queued_entered.wait(timeout=1)
            deadline = time.monotonic() + 3
            while service.health()["active_requests"] != 2:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            waiting = _assert_state(
                server, active=2, running=1, queued=1, draining=False
            )
            assert waiting["ai_journey_serving_queue_wait_seconds_count"] == 1
            clock[0] = 5.0
            cancelled.set()
            output = queued.result(timeout=1)
            assert output["finish_reason"] == "cancelled" and output["tokens"] == []
            assert bucket.tokens == 4
            after_cancel = _assert_state(
                server, active=1, running=1, queued=0, draining=False
            )
            assert after_cancel["ai_journey_serving_queue_wait_seconds_count"] == 2
            assert after_cancel["ai_journey_serving_queue_wait_seconds_sum"] == 3
            assert not service.drain(timeout_seconds=0)
            _assert_state(server, active=1, running=1, queued=0, draining=True)
        finally:
            clock[0] = 7.0
            cancelled.set()
            release.set()
        assert len(active.result(timeout=3)["tokens"]) == 2
        assert service.drain(timeout_seconds=1)
        final = _assert_state(server, active=0, running=0, queued=0, draining=True)
        assert final["ai_journey_serving_generated_tokens_total"] == 2
        assert final["ai_journey_serving_queue_wait_seconds_count"] == 2
        assert final["ai_journey_serving_queue_wait_seconds_sum"] == 3
        with pytest.raises(ServiceDraining):
            service.generate(request("private-rejected"))
        text = _get(server, "/metrics")
        for private in (
            "private-active",
            "private-queued",
            "private-rejected",
            service.model_sha256,
        ):
            assert private not in text
        assert bucket.tokens == 4
