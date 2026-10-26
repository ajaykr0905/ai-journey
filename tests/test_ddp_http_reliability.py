from contextlib import contextmanager
import http.client
import json
import threading

import pytest

from ai_journey.ddp_http import make_server


@contextmanager
def serving(service):
    server = make_server(service)
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


class FailingService:
    def __init__(self, after_token=False):
        self.after_token = after_token

    def health(self):
        return {"ready": True}

    def generate(self, request, *, on_token=None, **kwargs):
        if self.after_token and on_token is not None:
            on_token(0, 1)
        raise RuntimeError("internal diagnostic must remain server-side")


class RecordingService:
    def __init__(self):
        self.calls = 0

    def health(self):
        return {"ready": True}

    def generate(self, request, **kwargs):
        self.calls += 1
        return {"tokens": [], "finish_reason": "length"}


@pytest.mark.parametrize("route", ["/generate", "/stream"])
def test_model_failure_returns_safe_http_error_and_preserves_server(route):
    with serving(FailingService()) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        client.request(
            "POST",
            route,
            body=json.dumps({"request_id": "failure", "prompt": [1], "sampling": {}}),
        )
        response = client.getresponse()
        assert response.status == 500
        assert json.loads(response.read()) == {"error": "generation failed"}
        client.request("GET", "/health")
        assert client.getresponse().status == 200
        client.close()


def test_model_failure_after_stream_start_emits_one_terminal_error():
    with serving(FailingService(after_token=True)) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        client.request(
            "POST",
            "/stream",
            body=json.dumps({"request_id": "failure", "prompt": [1], "sampling": {}}),
        )
        response = client.getresponse()
        body = response.read().decode()
        assert response.status == 200
        assert body.count("event: token") == 1
        assert body.count("event: error") == 1
        assert "event: done" not in body
        assert "internal diagnostic" not in body
        client.request("GET", "/health")
        assert client.getresponse().status == 200
        client.close()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"request_id":"one","request_id":"two","prompt":[1],"sampling":{}}',
        b'{"request_id":"one","prompt":[1],"sampling":{"temperature":0,"temperature":1}}',
    ],
)
def test_duplicate_json_fields_are_rejected_before_service_execution(payload):
    service = RecordingService()
    with serving(service) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        client.request("POST", "/generate", body=payload)
        response = client.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "invalid generation request"}
        assert service.calls == 0
        client.close()
