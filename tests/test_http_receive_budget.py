import http.client
import json
import socket
import threading
import time

import pytest

from ai_journey.ddp_http import make_server
from test_ddp_http_reliability import RecordingService


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_trickle_input_cannot_renew_the_absolute_receive_budget(phase):
    service = RecordingService()
    server = make_server(service, max_connections=1, receive_timeout_seconds=0.08)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    client = socket.create_connection(server.server_address, timeout=2)
    stopped = threading.Event()

    def trickle():
        # Read the response concurrently: continuing to write after the server
        # has sent 408 and closed can make TCP reset unread response bytes.
        for _ in range(6):
            if stopped.wait(0.02):
                return
            try:
                client.sendall(b" ")
            except OSError:
                return

    writer = threading.Thread(target=trickle)
    try:
        client.sendall(
            b"POST /generate HTTP/1.0\r\nContent-Length: 100\r\n"
            + (b"X-Slow: " if phase == "headers" else b"\r\n{")
        )
        # Every received byte stays below the idle bound, while total receipt
        # exceeds the declared absolute budget.
        started = time.monotonic()
        writer.start()
        response = http.client.HTTPResponse(client)
        response.begin()
        assert response.status == 408
        assert json.loads(response.read()) == {
            "error": "request receive deadline exceeded"
        }
        stopped.set()
        writer.join(timeout=1)
        assert not writer.is_alive()
        assert time.monotonic() - started < 1
        assert service.calls == 0
        assert server._slots.acquire(timeout=0.5)
        server._slots.release()
        health = http.client.HTTPConnection(*server.server_address, timeout=1)
        health.request("GET", "/health")
        response = health.getresponse()
        assert response.status == 200
        response.read()
        health.close()
    finally:
        stopped.set()
        if writer.ident is not None:
            writer.join(timeout=1)
        client.close()
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), 31])
def test_receive_budget_controls_reject_before_socket_allocation(timeout):
    with pytest.raises(ValueError, match="receive"):
        make_server(RecordingService(), receive_timeout_seconds=timeout)
