from contextlib import contextmanager
import socket
import threading

import pytest

from ai_journey.ddp_http import make_server
from ai_journey.ddp_service import GenerationService, TokenBucket
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.http_measurements import (
    loopback_address,
    measure_http_generation,
    measure_http_output_parity,
)


@contextmanager
def running_service(**kwargs):
    service = GenerationService(GPT2LanguageModel(GPT2Config(8, 4, 8, 2, 1)), **kwargs)
    server = make_server(service)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", service
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def request(identifier="request"):
    return {
        "request_id": identifier,
        "prompt": [1, 2],
        "sampling": {"max_new_tokens": 3, "temperature": 0, "seed": 7},
    }


def test_real_http_measures_token_receipt_and_fresh_id_output_parity():
    with running_service() as (endpoint, service):
        report = measure_http_output_parity(endpoint, request())
    assert report["outputs_match"]
    streamed, plain = report["stream"], report["json"]
    assert streamed["request_id"] != plain["request_id"]
    assert service.completed_requests == 2
    assert len(streamed["token_received_ns"]) == 3
    assert streamed["token_received_ns"] == sorted(streamed["token_received_ns"])
    assert (
        streamed["started_ns"]
        <= streamed["headers_ns"]
        <= streamed["token_received_ns"][0]
        <= streamed["finished_ns"]
    )
    assert not plain["token_received_ns"]


def test_actual_admission_rejections_and_client_cancellation_are_distinct():
    with running_service(admission=TokenBucket(5, 0.00001)) as (endpoint, service):
        first = measure_http_generation(endpoint, request("first"))
        second = measure_http_generation(endpoint, request("second"))
    assert first["outcome"] == "completed"
    assert second["status"] == 429 and second["outcome"] == "rejected"
    with running_service() as (endpoint, service):
        cancelled = measure_http_generation(endpoint, request(), cancel_after_tokens=1)
    assert cancelled["outcome"] == "cancelled"
    assert len(cancelled["tokens"]) == len(cancelled["token_received_ns"]) == 1


def test_connection_failure_is_recorded_and_external_addresses_are_rejected():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    result = measure_http_generation(f"http://127.0.0.1:{port}", request())
    assert result["outcome"] == "transport_error"
    assert result["finished_ns"] >= result["started_ns"]
    for endpoint in (
        "https://127.0.0.1:80",
        "http://localhost:80",
        "http://user@127.0.0.1:80",
        "http://127.0.0.1:80/path",
    ):
        with pytest.raises(ValueError, match="endpoint"):
            loopback_address(endpoint)
