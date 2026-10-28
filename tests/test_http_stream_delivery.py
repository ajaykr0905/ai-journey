from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from ai_journey.http_measurements import measure_http_generation
from ai_journey.http_stream import GenerationEventDecoder, StreamProtocolError
import ai_journey.http_measurements as measurements
from test_http_measurements import request

TOKEN = b'event: token\ndata: {"index":0,"token":3}\n\n'
BAD_EVENT = b"event: unknown\ndata: {}\n\n"


class FakeSocket:
    def shutdown(self, how):
        pass


class FakeResponse:
    status = 200
    length = None

    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def getheader(self, name, default):
        return "text/event-stream"

    def read1(self, maximum):
        return next(self.chunks, b"")

    def close(self):
        pass


class FakeConnection:
    def __init__(self, chunks):
        self.sock = FakeSocket()
        self.response = FakeResponse(chunks)

    def connect(self):
        pass

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        pass


@pytest.mark.parametrize(
    "suffix",
    [BAD_EVENT, b"\xff", b"x" * (16 * 1024 + 1)],
    ids=["unknown-event", "invalid-utf8", "event-budget"],
)
@pytest.mark.parametrize("cancel_after_tokens", [None, 1])
def test_partial_delivery_and_planned_cancellation_ignore_read_coalescing(
    monkeypatch, suffix, cancel_after_tokens
):
    payload = TOKEN + suffix
    observations = []
    for chunks in ([payload], [TOKEN, suffix], [bytes([byte]) for byte in payload]):
        connection = FakeConnection(chunks)
        monkeypatch.setattr(
            measurements.http.client,
            "HTTPConnection",
            lambda *args, **kwargs: connection,
        )
        observed = measure_http_generation(
            "http://127.0.0.1:12345", request(), cancel_after_tokens=cancel_after_tokens
        )
        observations.append(observed)
        assert observed["tokens"] == [3]
        assert len(observed["token_received_ns"]) == 1
        assert observed["outcome"] == (
            "cancelled" if cancel_after_tokens else "protocol_error"
        )
        assert observed["error_kind"] == (
            "client_closed_stream" if cancel_after_tokens else "StreamProtocolError"
        )
        assert observed["model_sha256"] is observed["finish_reason"] is None
    assert (
        len(
            {
                (row["outcome"], tuple(row["tokens"]), row["error_kind"])
                for row in observations
            }
        )
        == 1
    )


def test_lazy_decoder_yields_the_valid_prefix_before_inspecting_a_later_error():
    decoder = GenerationEventDecoder("request", prompt_tokens=2, max_tokens=3)
    events = decoder.iter_events(TOKEN + BAD_EVENT)
    assert next(events) == ("token", {"index": 0, "token": 3})
    assert decoder.tokens == [3]
    with pytest.raises(StreamProtocolError, match="unsupported"):
        next(events)


@contextmanager
def malformed_peer():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(TOKEN + BAD_EVENT)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("cancel_after_tokens", [None, 1])
def test_actual_socket_retains_accepted_token_before_a_later_bad_frame(
    cancel_after_tokens,
):
    with malformed_peer() as endpoint:
        observed = measure_http_generation(
            endpoint, request(), cancel_after_tokens=cancel_after_tokens
        )
    assert observed["tokens"] == [3]
    assert len(observed["token_received_ns"]) == 1
    assert observed["outcome"] == (
        "cancelled" if cancel_after_tokens else "protocol_error"
    )
