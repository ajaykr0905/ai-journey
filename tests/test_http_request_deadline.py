from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from ai_journey.http_measurements import measure_http_generation
from test_http_measurements import request


@contextmanager
def stalled_peer(*, after_headers):
    entered = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            if after_headers:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
            # A genuine local peer stall, not a fabricated timing observation.
            time.sleep(0.4)
            if not after_headers:
                try:
                    self.send_response(200)
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("after_headers", [False, True])
def test_whole_request_deadline_interrupts_actual_header_and_body_stalls(after_headers):
    with stalled_peer(after_headers=after_headers) as (endpoint, entered):
        measured = measure_http_generation(endpoint, request(), timeout_seconds=0.03)
        assert entered.is_set()
    assert measured["outcome"] == "deadline"
    assert measured["error_kind"] == "client_request_deadline"
    assert 0 < measured["finished_ns"] - measured["started_ns"] < 300_000_000
    assert not measured["tokens"]
