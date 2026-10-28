from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import http.client
import json
import threading
import time

import pytest

from ai_journey.ddp_generation import SamplingConfig
from ai_journey.ddp_service import GenerationRequest, GenerationService, TokenBucket
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from test_ddp_metrics_integration import server_for


@pytest.mark.parametrize("handoff", ["cancelled", "valid", "callback-error"])
def test_decode_handoff_preserves_budget_and_capacity_for_fresh_work(handoff):
    bucket = TokenBucket(6 if handoff == "valid" else 4, 1, clock=lambda: 0.0)
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)),
        admission=bucket,
        max_inflight_requests=2,
    )
    active_entered, active_release, first_read, allow_first_return, cancelled = (
        threading.Event() for _ in range(5)
    )
    request = GenerationRequest(
        "active", (1,), SamplingConfig(max_new_tokens=1, temperature=0)
    )

    def pause_active(index, token):
        active_entered.set()
        assert active_release.wait(timeout=5)

    def queued_cancellation():
        observed = cancelled.is_set()
        if not first_read.is_set():
            first_read.set()
            assert allow_first_return.wait(timeout=5)
        if observed and handoff == "callback-error":
            raise RuntimeError("cancellation callback failed")
        return observed

    with ThreadPoolExecutor(max_workers=2) as workers:
        active = workers.submit(service.generate, request, on_token=pause_active)
        try:
            assert active_entered.wait(timeout=5)
            queued = workers.submit(
                service.generate,
                replace(request, request_id="queued"),
                cancelled=queued_cancellation,
            )
            assert first_read.wait(timeout=5)
            assert service.health()["queued_requests"] == 1
            if handoff != "valid":
                cancelled.set()
            active_release.set()
            assert active.result(timeout=5)["finish_reason"] == "length"
        finally:
            active_release.set()
            allow_first_return.set()
        if handoff == "callback-error":
            with pytest.raises(RuntimeError, match="cancellation callback failed"):
                queued.result(timeout=5)
        else:
            output = queued.result(timeout=5)
            assert output["finish_reason"] == (
                "length" if handoff == "valid" else "cancelled"
            )
            assert len(output["tokens"]) == (1 if handoff == "valid" else 0)

    assert bucket.tokens == 2
    fresh = service.generate(
        replace(request, request_id="fresh"), deadline=time.monotonic() + 5
    )
    assert fresh["finish_reason"] == "length"
    assert service.health()["active_requests"] == 0
    assert service.health()["completed_requests"] == (3 if handoff == "valid" else 2)
    assert service.metrics.snapshot()["requests"] == {
        "success": 3 if handoff == "valid" else 2,
        "cancelled": int(handoff == "cancelled"),
        "deadline": 0,
        "error": int(handoff == "callback-error"),
        "rejected": 0,
    }


@pytest.mark.parametrize("control", ["deadline", "cancelled"])
def test_queued_interruption_finishes_without_waiting_for_active_decode(
    monkeypatch, control
):
    bucket = TokenBucket(100, 1, clock=lambda: 0.0)
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), admission=bucket
    )
    request = GenerationRequest("active", (1,), SamplingConfig(max_new_tokens=2))
    entered, release, finished = (threading.Event() for _ in range(3))
    cancelled = threading.Event()
    results = []

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return {"tokens": [1, 2], "prompt_tokens": 1, "finish_reason": "length"}

    monkeypatch.setattr("ai_journey.ddp_service.generate_tokens", blocked)
    active = threading.Thread(target=service.generate, args=(request,))
    active.start()

    def queued_request():
        controls = (
            {"deadline": time.monotonic() + 0.03}
            if control == "deadline"
            else {"cancelled": cancelled.is_set}
        )
        results.append(
            service.generate(replace(request, request_id="queued"), **controls)
        )
        finished.set()

    queued = threading.Thread(target=queued_request)
    try:
        assert entered.wait(timeout=1)
        charged = bucket.tokens
        queued.start()
        cancelled.set()
        assert finished.wait(
            timeout=0.5
        ), "queued controls must not await the decode lock"
        assert results[0]["finish_reason"] == control
        assert results[0]["tokens"] == []
        assert bucket.tokens == charged
        assert service.health()["active_requests"] == 1
    finally:
        release.set()
        active.join(timeout=3)
        if queued.ident is not None:
            queued.join(timeout=3)
    assert not active.is_alive() and not queued.is_alive()
    assert service.health()["active_requests"] == 0


@pytest.mark.parametrize("limit", [True, 0, 65, 1.5])
def test_inference_capacity_rejects_invalid_controls_before_model_copy(limit):
    with pytest.raises(ValueError):
        GenerationService(object(), max_inflight_requests=limit)


def test_http_inference_overload_rejects_without_leaking_capacity(monkeypatch):
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), max_inflight_requests=1
    )
    entered, release = threading.Event(), threading.Event()
    outcomes = []

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return {"tokens": [1], "prompt_tokens": 1, "finish_reason": "length"}

    monkeypatch.setattr("ai_journey.ddp_service.generate_tokens", blocked)
    payload = json.dumps(
        {"request_id": "inflight", "prompt": [1], "sampling": {"max_new_tokens": 1}}
    )
    with server_for(service) as server:

        def post_active():
            client = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                client.request("POST", "/generate", body=payload)
                response = client.getresponse()
                outcomes.append(response.status)
                response.read()
            finally:
                client.close()

        active = threading.Thread(target=post_active)
        active.start()
        try:
            assert entered.wait(timeout=1)
            client = http.client.HTTPConnection(*server.server_address, timeout=2)
            client.request("POST", "/generate", body=payload)
            response = client.getresponse()
            assert response.status == 503
            assert response.getheader("Retry-After") == "1"
            assert json.loads(response.read()) == {"error": "inference queue is full"}
            client.close()
            assert service.health()["active_requests"] == 1
        finally:
            release.set()
            active.join(timeout=3)
        assert outcomes == [200]
        assert service.health()["active_requests"] == 0
        # A complete cached response can enter again after the active slot exits.
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        client.request("POST", "/generate", body=payload)
        response = client.getresponse()
        assert response.status == 200
        response.read()
        client.close()
    assert service.metrics.snapshot()["requests"]["rejected"] == 1
