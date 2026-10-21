from contextlib import contextmanager
import http.client
import json
import threading

from ai_journey.ddp_http import make_server
from ai_journey.ddp_service import GenerationService
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_config import GPT2Config


@contextmanager
def running_server():
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)))
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


def test_real_loopback_http_generates_and_streams_ordered_tokens():
    with running_server() as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        client.request("GET", "/health")
        response = client.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["ready"]
        payload = json.dumps(
            {
                "request_id": "http-test",
                "prompt": [1, 2],
                "sampling": {"temperature": 0, "max_new_tokens": 3},
            }
        )
        client.request("POST", "/stream", body=payload)
        response = client.getresponse()
        body = response.read().decode()
        assert response.status == 200
        assert body.count("event: token") == 3
        assert body.count("event: done") == 1
        client.close()


def test_http_rejects_invalid_json_without_exposing_internal_details():
    with running_server() as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=5)
        client.request("POST", "/generate", body=b"not-json")
        response = client.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "invalid generation request"}
        client.close()
