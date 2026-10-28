"""Generation terminals cannot turn premature HTTP body EOF into successful work."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from ai_journey.http_load import run_http_load
from ai_journey.http_load_cli import main
from ai_journey.http_load_profile import HTTPWorkloadProfile
from ai_journey.http_load_report import build_http_load_report, verify_http_load_report


@contextmanager
def framing_peer(framing="short-length", *, hold=False):
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            output = {
                "request_id": request["request_id"],
                "model_sha256": "a" * 64,
                "prompt_tokens": len(request["prompt"]),
                "tokens": [3, 4],
                "finish_reason": "length",
            }
            frames = [
                ("token", {"index": 0, "token": 3}),
                ("token", {"index": 1, "token": 4}),
                ("done", output),
            ]
            streaming = self.path == "/stream"
            body = (
                "".join(
                    f"event: {event}\ndata: {json.dumps(value)}\n\n"
                    for event, value in frames
                ).encode()
                if streaming
                else json.dumps(output).encode()
            )
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            chunked = framing in {"chunked", "short-chunk", "missing-chunk-end"}
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            elif framing != "close":
                self.send_header(
                    "Content-Length",
                    str(len(body) + (7 if framing == "short-length" else 0)),
                )
            self.send_header("Connection", "close")
            self.end_headers()
            if chunked:
                length = len(body) + (7 if framing == "short-chunk" else 0)
                self.wfile.write(f"{length:x}\r\n".encode() + body)
                if framing != "short-chunk":
                    self.wfile.write(b"\r\n")
                if framing == "chunked":
                    self.wfile.write(b"0\r\n\r\n")
            else:
                self.wfile.write(body)
            self.wfile.flush()
            if hold:
                release.wait(timeout=1)
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("streaming", [True, False], ids=["sse", "json"])
@pytest.mark.parametrize(
    "framing", ["short-length", "short-chunk", "missing-chunk-end"]
)
def test_premature_body_eof_does_not_count_a_valid_terminal_as_completed_work(
    streaming, framing
):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=2,
        timeout_seconds=0.5,
        streaming=streaming,
    )
    with framing_peer(framing) as endpoint:
        raw = run_http_load(endpoint, profile)

    observation = raw["measured"][0]["observation"]
    assert observation["outcome"] == "transport_error"
    assert observation["error_kind"] == "IncompleteRead"
    assert observation["tokens"] == ([3, 4] if streaming else [])
    assert observation["model_sha256"] is observation["finish_reason"] is None
    assert raw["totals"]["outcomes"]["completed"] == 0
    assert raw["totals"]["completed_output_tokens"] == 0
    assert raw["totals"]["partial_observed_tokens"] == (2 if streaming else 0)
    verify_http_load_report(build_http_load_report(raw))


@pytest.mark.parametrize("streaming", [True, False], ids=["sse", "json"])
@pytest.mark.parametrize("framing", ["length", "chunked", "close"])
def test_complete_http_framing_preserves_success_and_verified_token_totals(
    streaming, framing
):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=2,
        streaming=streaming,
    )
    with framing_peer(framing) as endpoint:
        raw = run_http_load(endpoint, profile)

    observation = raw["measured"][0]["observation"]
    assert observation["outcome"] == "completed"
    assert observation["tokens"] == [3, 4]
    assert observation["error_kind"] is None
    assert raw["totals"]["outcomes"]["completed"] == 1
    assert raw["totals"]["completed_output_tokens"] == 2
    verify_http_load_report(build_http_load_report(raw))


@pytest.mark.parametrize(
    "framing", ["short-length", "short-chunk", "missing-chunk-end"]
)
def test_intentional_token_boundary_cancellation_precedes_http_completion(framing):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=2,
        cancel_every=1,
        cancel_after_tokens=1,
    )
    with framing_peer(framing) as endpoint:
        raw = run_http_load(endpoint, profile)

    observation = raw["measured"][0]["observation"]
    assert observation["outcome"] == "cancelled"
    assert observation["error_kind"] == "client_closed_stream"
    assert observation["tokens"] == [3]
    assert observation["model_sha256"] is observation["finish_reason"] is None
    assert raw["totals"]["completed_output_tokens"] == 0
    assert raw["totals"]["partial_observed_tokens"] == 1
    verify_http_load_report(build_http_load_report(raw))


@pytest.mark.parametrize("streaming", [True, False], ids=["sse", "json"])
def test_deadline_still_overrides_an_unfinished_body_with_a_valid_terminal(streaming):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=2,
        timeout_seconds=0.1,
        streaming=streaming,
    )
    with framing_peer(hold=True) as endpoint:
        raw = run_http_load(endpoint, profile)

    observation = raw["measured"][0]["observation"]
    assert observation["outcome"] == "deadline"
    assert observation["error_kind"] == "client_request_deadline"
    assert observation["tokens"] == ([3, 4] if streaming else [])
    assert observation["model_sha256"] is observation["finish_reason"] is None
    assert observation["finished_ns"] - observation["started_ns"] < 800_000_000
    assert raw["totals"]["completed_output_tokens"] == 0
    verify_http_load_report(build_http_load_report(raw))


@pytest.mark.parametrize("streaming", [True, False], ids=["sse", "json"])
def test_cli_retains_and_verifies_premature_eof_without_successful_work(
    tmp_path, capsys, streaming
):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=2,
        streaming=streaming,
    )
    profile_path, output = tmp_path / "profile.json", tmp_path / "report.json"
    profile_path.write_text(json.dumps(profile.to_mapping()))
    with framing_peer() as endpoint:
        assert (
            main(
                [
                    "measure",
                    "--endpoint",
                    endpoint,
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    assert "completed=0" in capsys.readouterr().out
    assert main(["verify", "--report", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["measurement"]["totals"]["outcomes"]["transport_error"] == 1
    assert report["measurement"]["totals"]["completed_output_tokens"] == 0
