import re
import socket
import struct
import threading
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest
from test_serving_recovery_config import valid_config

from ai_journey.serving_recovery_client import (
    AttemptRecord,
    attempt_from_mapping,
    plan_requests,
    send_generation_attempt,
)
from ai_journey.serving_recovery_config import canonical_json_bytes
from ai_journey.serving_recovery_identity import server_lifetime_id, serving_run_id
from ai_journey.serving_recovery_journal import ExecutionJournal, read_execution_journal
from ai_journey.serving_recovery_model import build_serving_model, completion_body
from ai_journey.serving_recovery_server import RecoveryHTTPServer

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "day-19-demo-names.txt"
MODEL = build_serving_model(FIXTURE.read_bytes())
RUN = serving_run_id(valid_config(), model_sha256=MODEL.model_sha256)
LIFETIME = server_lifetime_id(RUN, 1)
PLAN = plan_requests(valid_config(), run_id=RUN, model=MODEL)


@contextmanager
def live_server(tmp_path):
    journal = ExecutionJournal(
        tmp_path / "journal.jsonl",
        lifetime_id=LIFETIME,
        model_sha256=MODEL.model_sha256,
        max_records=16,
    )
    server = RecoveryHTTPServer(
        model=MODEL, journal=journal, lifetime_id=LIFETIME, max_new_tokens=8
    )
    thread = threading.Thread(target=server.serve_forever, args=(0.01,))
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        journal.close()


@contextmanager
def peer(action):
    """Accept one request on loopback and answer with a scripted behavior."""

    listener = socket.create_server(("127.0.0.1", 0))
    release = threading.Event()

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(5)
            data = b""
            while b"\r\n\r\n" not in data:
                data += connection.recv(4096)
            head, _, body = data.partition(b"\r\n\r\n")
            length = int(re.search(rb"Content-Length: ([0-9]+)", head).group(1))
            while len(body) < length:
                body += connection.recv(4096)
            action(connection, release)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        release.set()
        listener.close()
        thread.join(timeout=10)


def attempt(port, *, deadline_ms=1000, number=1, planned=PLAN[0]):
    return send_generation_attempt(
        f"http://127.0.0.1:{port}",
        planned,
        attempt=number,
        deadline_ms=deadline_ms,
        model_sha256=MODEL.model_sha256,
    )


def respond(payload):
    return lambda connection, release: connection.sendall(payload)


def ok_body(planned=PLAN[0]):
    return completion_body(MODEL, planned.request)


def response(body, *, status=b"200 OK", lifetime=LIFETIME.encode(), length=None):
    length = len(body) if length is None else length
    headers = [b"HTTP/1.0 " + status, b"Content-Type: application/json"]
    headers.append(b"Content-Length: " + str(length).encode())
    if lifetime is not None:
        headers.append(b"X-Server-Lifetime: " + lifetime)
    return b"\r\n".join(headers) + b"\r\n\r\n" + body


def reset(connection, release):
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def test_plan_is_deterministic_and_keys_every_request():
    assert plan_requests(valid_config(), run_id=RUN, model=MODEL) == PLAN
    assert [planned.index for planned in PLAN] == list(range(24))
    assert len({planned.idempotency_key for planned in PLAN}) == 24
    assert {planned.request.max_new_tokens for planned in PLAN} == {8}
    assert all(set(p.request.prompt) <= set(MODEL.vocabulary[1:]) for p in PLAN)


def test_ok_attempt_carries_key_and_attempt_and_digests_exact_bytes(tmp_path):
    with live_server(tmp_path) as port:
        record = attempt(port, number=3, planned=PLAN[5])
    assert (record.outcome, record.http_status, record.error) == ("ok", 200, None)
    assert record.lifetime_id == LIFETIME
    assert record.response_sha256 == sha256(ok_body(PLAN[5])).hexdigest()
    assert record.finished_ns >= record.started_ns
    (execution,) = read_execution_journal(
        tmp_path / "journal.jsonl",
        lifetime_id=LIFETIME,
        max_records=16,
        allow_torn_tail=False,
    ).records
    assert (execution.idempotency_key, execution.attempt) == (
        PLAN[5].idempotency_key,
        3,
    )
    assert execution.response_sha256 == record.response_sha256


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize(
    ("action", "outcome", "error"),
    [
        (reset, "connection_reset", "ConnectionResetError"),
        (
            lambda connection, release: None,
            "connection_reset",
            "closed_before_response",
        ),
        (
            respond(response(ok_body(), length=len(ok_body()) + 5)),
            "connection_reset",
            "truncated_body",
        ),
        (lambda connection, release: release.wait(5), "timeout", "receive_timeout"),
        (
            respond(response(b"{}", status=b"503 Service Unavailable")),
            "http_error",
            "status_503",
        ),
        (respond(b"SPDY/9 200 OK\r\n\r\n"), "invalid_response", "status_line"),
        (
            respond(response(ok_body(), lifetime=None)),
            "invalid_response",
            "response_body",
        ),
        (
            respond(response(ok_body().replace(b":", b": ", 1))),
            "invalid_response",
            "response_body",
        ),
        (
            respond(response(ok_body(PLAN[1]))),
            "invalid_response",
            "response_body",
        ),
        (respond(response(ok_body(), length=1)), "invalid_response", "excess_body"),
    ],
)
def test_failed_attempts_have_exactly_one_class(action, outcome, error):
    with peer(action) as port:
        record = attempt(port, deadline_ms=200)
    assert (record.outcome, record.error) == (outcome, error)
    assert (record.lifetime_id, record.response_sha256) == (None, None)
    assert record.port == port


def test_refused_connections_are_classified_before_any_exchange():
    record = attempt(closed_port())
    assert (record.outcome, record.error) == (
        "connection_refused",
        "ConnectionRefusedError",
    )


def test_attempt_records_round_trip_and_reject_incoherent_evidence(tmp_path):
    with live_server(tmp_path) as port:
        record = attempt(port)
    assert attempt_from_mapping(record.to_mapping()) == record
    assert canonical_json_bytes(record.to_mapping())
    for changes, message in (
        ({"outcome": "lost"}, "unsupported"),
        ({"finished_ns": record.started_ns - 1}, "finished_ns"),
        ({"http_status": 500}, "clean 200"),
        ({"outcome": "timeout"}, "no served identity"),
        ({"request_index": 1}, "another request"),
        ({"attempt": True}, "integer"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            AttemptRecord(**(record.to_mapping() | changes))
    with pytest.raises(ValueError, match="fields"):
        attempt_from_mapping(record.to_mapping() | {"extra": 1})
    with pytest.raises(ValueError, match="endpoint"):
        send_generation_attempt(
            "http://localhost:80",
            PLAN[0],
            attempt=1,
            deadline_ms=10,
            model_sha256=MODEL.model_sha256,
        )
