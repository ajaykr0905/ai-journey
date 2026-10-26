import http.client
import json

import pytest

from ai_journey.ddp_http import make_server
from ai_journey.ddp_service import GenerationRequest, GenerationService, TokenBucket
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.http_measurements import measure_http_generation
from test_ddp_http_reliability import RecordingService, serving


@pytest.mark.parametrize("route", ["/generate", "/stream"])
@pytest.mark.parametrize("field", ["temperature", "top_p"])
@pytest.mark.parametrize("sign", [1, -1])
def test_unrepresentable_numbers_return_400_without_charging_or_leaking_capacity(
    route, field, sign
):
    bucket = TokenBucket(8, 1, clock=lambda: 0.0)
    service = GenerationService(
        GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), admission=bucket
    )
    errors = []
    with serving(service, max_connections=1) as server:
        server.handle_error = lambda *args: errors.append("unhandled handler error")
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            payload = {
                "request_id": "oversized",
                "prompt": [1],
                "sampling": {field: sign * 10**1000},
            }
            body = json.dumps(payload)
            assert len(body) < 16 * 1024
            client.request("POST", route, body=body)
            response = client.getresponse()
            assert response.status == 400
            assert json.loads(response.read()) == {
                "error": "invalid generation request"
            }
            assert not errors
            assert service.completed_requests == 0
            assert service.health()["active_requests"] == 0
            assert bucket.tokens == 8
            assert server._slots.acquire(timeout=1)
            server._slots.release()
            payload["request_id"] = "valid"
            payload["sampling"] = {"temperature": 0, "max_new_tokens": 1}
            client.request("POST", route, body=json.dumps(payload))
            response = client.getresponse()
            assert response.status == 200
            output = response.read()
            if route == "/stream":
                assert b"event: done" in output
            else:
                assert len(json.loads(output)["tokens"]) == 1
            assert service.completed_requests == 1
            assert bucket.tokens == 6
        finally:
            client.close()


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_serialized_service_request_reports_numeric_overflow_as_invalid(field):
    with pytest.raises(ValueError, match="sampling"):
        GenerationRequest.from_mapping(
            {"request_id": "large", "prompt": [1], "sampling": {field: 10**1000}}
        )


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_measurement_client_rejects_numeric_overflow_before_network(field, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid numeric input must not open a connection")

    monkeypatch.setattr(http.client, "HTTPConnection", unexpected)
    with pytest.raises(ValueError, match="benchmark"):
        measure_http_generation(
            "http://127.0.0.1:1",
            {"request_id": "large", "prompt": [1], "sampling": {field: 10**1000}},
        )


@pytest.mark.parametrize(
    "timeout", [10**1000, -(10**1000)], ids=["positive", "negative"]
)
def test_unrepresentable_receive_budget_is_rejected_before_socket_creation(timeout):
    with pytest.raises(ValueError, match="receive"):
        make_server(RecordingService(), receive_timeout_seconds=timeout)
