import http.client
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.serving_recovery_identity import idempotency_key, server_lifetime_id
from ai_journey.serving_recovery_journal import read_execution_journal
from ai_journey.serving_recovery_model import (
    GenerationRequest,
    build_serving_model,
    completion_body,
)
from ai_journey.serving_recovery_server import (
    ServerHandshake,
    health_body,
    parse_handshake,
    server_command,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "day-19-demo-names.txt"
RUN = "serving-recovery-test-0123456789abcdef"
LIFETIME = server_lifetime_id(RUN, 1)
MODEL = build_serving_model(FIXTURE.read_bytes())


def command(tmp_path, **changes):
    options = {
        "source_path": FIXTURE,
        "journal_path": tmp_path / "journal.jsonl",
        "lifetime_id": LIFETIME,
        "model_sha256": MODEL.model_sha256,
        "max_new_tokens": 8,
        "max_records": 32,
    }
    return server_command(**(options | changes))


def read_line(process, timeout=10.0):
    data = b""
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while not data.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            assert remaining > 0, "server handshake timed out"
            if selector.select(remaining):
                chunk = os.read(process.stdout.fileno(), 1024)
                if not chunk:
                    break
                data += chunk
    return data


@contextmanager
def running(argv, stderr=subprocess.DEVNULL):
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        process.stdout.close()


def ready(process):
    return parse_handshake(
        read_line(process),
        lifetime_id=LIFETIME,
        pid=process.pid,
        model_sha256=MODEL.model_sha256,
    )


def exchange(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def generate(port, request, *, index=0, attempt=1, headers=None):
    values = {
        "Content-Type": "application/json",
        "Idempotency-Key": idempotency_key(RUN, index),
        "X-Attempt": str(attempt),
    }
    return exchange(
        port, "POST", "/generate", request.body_bytes(), values | (headers or {})
    )


def journal(tmp_path, *, allow_torn_tail=False):
    return read_execution_journal(
        tmp_path / "journal.jsonl",
        lifetime_id=LIFETIME,
        max_records=32,
        allow_torn_tail=allow_torn_tail,
    )


def test_server_command_is_isolated_shell_free_and_validated(tmp_path):
    argv = command(tmp_path)

    assert argv[:3] == (sys.executable, "-I", "-c")
    assert Path(argv[4]) == Path(__file__).resolve().parents[1] / "src" / "ai_journey"
    assert argv[argv.index("--lifetime-id") + 1] == LIFETIME
    assert argv[-2:] == ("--hold-response-ms", "0")
    with pytest.raises(ValueError, match="hold_response_ms"):
        command(tmp_path, hold_response_ms=10_001)
    with pytest.raises(ValueError, match="lifetime_id"):
        command(tmp_path, lifetime_id="lifetime-x")
    with pytest.raises(TypeError, match="pathlib"):
        command(tmp_path, journal_path="journal.jsonl")


def test_server_handshakes_serves_health_and_journals_before_responding(tmp_path):
    with running(command(tmp_path)) as process:
        handshake = ready(process)
        status, headers, body = exchange(handshake.port, "GET", "/health")
        assert status == 200
        assert body == health_body(
            lifetime_id=LIFETIME, pid=process.pid, model_sha256=MODEL.model_sha256
        )
        assert headers["X-Server-Lifetime"] == LIFETIME

        request = GenerationRequest("ka", 7, 8)
        status, headers, body = generate(handshake.port, request, attempt=2)
        assert status == 200
        assert body == completion_body(MODEL, request)
        assert headers["X-Server-Lifetime"] == LIFETIME
        (record,) = journal(tmp_path).records
        assert record.idempotency_key == idempotency_key(RUN, 0)
        assert record.attempt == 2
        assert record.request_sha256 == request.request_sha256()
        assert record.response_sha256 == sha256(body).hexdigest()

        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    assert len(journal(tmp_path).records) == 1


@pytest.mark.parametrize(
    ("body", "headers", "status"),
    [
        (b'{"prompt":"a","seed":1}', {}, 400),
        (b'{"prompt":"b","seed":1,"max_new_tokens":2}', {}, 400),
        (b'{"prompt":"a","seed":1,"max_new_tokens":9}', {}, 400),
        (b'{"prompt":"a","seed":1,"seed":2,"max_new_tokens":2}', {}, 400),
        (None, {"X-Attempt": "0"}, 400),
        (None, {"X-Attempt": "01"}, 400),
        (None, {"Idempotency-Key": "client-chosen"}, 400),
    ],
)
def test_server_rejects_invalid_requests_without_executing_them(
    tmp_path, body, headers, status
):
    request = GenerationRequest("a", 1, 2)
    with running(command(tmp_path)) as process:
        port = ready(process).port
        values = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key(RUN, 0),
            "X-Attempt": "1",
        }
        got, _, payload = exchange(
            port, "POST", "/generate", body or request.body_bytes(), values | headers
        )
        assert (got, json.loads(payload)) == (
            status,
            {"error": "invalid generation request"},
        )
        assert exchange(port, "GET", "/missing")[0] == 404
        assert journal(tmp_path).records == ()


def test_sigkilled_server_leaves_a_readable_committed_journal(tmp_path):
    with running(command(tmp_path)) as process:
        port = ready(process).port
        for index in range(3):
            assert (
                generate(port, GenerationRequest("ma", index, 8), index=index)[0] == 200
            )
        os.killpg(process.pid, signal.SIGKILL)
        assert process.wait(timeout=10) == -signal.SIGKILL

    contents = journal(tmp_path, allow_torn_tail=True)
    assert [record.sequence for record in contents.records] == [0, 1, 2]
    assert contents.torn_tail_bytes == 0


def test_server_cold_start_imports_neither_torch_nor_numpy(tmp_path):
    argv = command(tmp_path)
    instrumented = (argv[0], "-X", "importtime", *argv[1:])
    log = tmp_path / "imports.log"
    with log.open("wb") as stderr, running(instrumented, stderr=stderr) as process:
        ready(process)
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 0

    imported = {
        line.rsplit("|", 1)[-1].strip()
        for line in log.read_text().splitlines()
        if line.startswith("import time:")
    }
    assert "ai_journey.serving_recovery_server" in imported
    assert not {name for name in imported if name.split(".")[0] in {"torch", "numpy"}}


def test_server_refuses_a_mismatched_model_or_an_existing_journal(tmp_path):
    with running(command(tmp_path, model_sha256="0" * 64)) as process:
        assert process.wait(timeout=10) == 2
        assert read_line(process) == b""

    (tmp_path / "journal.jsonl").write_bytes(b"")
    with running(command(tmp_path)) as process:
        assert process.wait(timeout=10) == 2


def test_handshake_parser_binds_child_lifetime_and_model():
    line = ServerHandshake(LIFETIME, 4242, 8080, MODEL.model_sha256).to_line()
    expected = {
        "lifetime_id": LIFETIME,
        "pid": 4242,
        "model_sha256": MODEL.model_sha256,
    }

    assert parse_handshake(line, **expected).port == 8080
    for changes, message in (
        ({"pid": 4243}, "spawned child"),
        ({"lifetime_id": server_lifetime_id(RUN, 2)}, "another server"),
        ({"model_sha256": "0" * 64}, "unexpected model"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_handshake(line, **(expected | changes))
    for bad, message in (
        (line[:-1], "newline"),
        (line.replace(b":8080", b": 8080"), "canonical"),
        (line.replace(b'"ready"', b'"busy"'), "readiness"),
        (line.replace(b"8080", b"0"), "port"),
        (line.replace(b'"pid"', b'"pid":1,"x"'), "fields"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_handshake(bad, **expected)
