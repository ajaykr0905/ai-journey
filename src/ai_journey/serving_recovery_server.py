"""Loopback inference server process for the serving recovery lab.

The server binds 127.0.0.1 on an ephemeral port and publishes its actual port,
pid, lifetime and model identity as one JSON handshake line on stdout.
``GET /health`` reports the same identity. ``POST /generate`` validates a
bounded request, computes the deterministic completion, commits an execution
journal record and only then sends the response. SIGTERM drains and exits 0.

``server_command`` starts the module in isolated mode (``python -I``) behind a
package stub, so the child imports only the standard library and this lab's
stdlib-only modules. The package ``__init__`` eagerly imports torch, which
would add more than a second to every cold restart on the measured host.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socketserver
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .serving_recovery_config import (
    ServingRecoveryError,
    canonical_json_bytes,
    read_bounded_file,
    require_digest,
    require_int,
    sha256_hex,
    strict_json_loads,
)
from .serving_recovery_identity import require_idempotency_key, require_lifetime_id
from .serving_recovery_journal import (
    MAX_ATTEMPT,
    MAX_JOURNAL_RECORDS,
    ExecutionJournal,
)
from .serving_recovery_model import (
    MAX_NEW_TOKENS,
    MAX_SOURCE_BYTES,
    ServingModel,
    build_serving_model,
    completion_body,
    request_from_mapping,
)

LOOPBACK = "127.0.0.1"
MAX_BODY_BYTES = 4096
MAX_HANDSHAKE_BYTES = 1024
MAX_CONNECTIONS = 16
SOCKET_TIMEOUT_SECONDS = 5
KEY_HEADER = "Idempotency-Key"
ATTEMPT_HEADER = "X-Attempt"
LIFETIME_HEADER = "X-Server-Lifetime"
_DECIMAL = re.compile(r"[1-9][0-9]{0,4}")
_BOOTSTRAP = (
    "import importlib.machinery, importlib.util, sys\n"
    "spec = importlib.machinery.ModuleSpec('ai_journey', None, is_package=True)\n"
    "spec.submodule_search_locations = [sys.argv[1]]\n"
    "sys.modules['ai_journey'] = importlib.util.module_from_spec(spec)\n"
    "from ai_journey.serving_recovery_server import main\n"
    "raise SystemExit(main(sys.argv[2:]))\n"
)


def server_command(
    *,
    source_path: Path,
    journal_path: Path,
    lifetime_id: str,
    model_sha256: str,
    max_new_tokens: int,
    max_records: int,
) -> tuple[str, ...]:
    """Build an isolated, shell-free argument vector for one server lifetime."""

    if not isinstance(source_path, Path) or not isinstance(journal_path, Path):
        raise TypeError("server paths must be pathlib.Path values")
    require_lifetime_id(lifetime_id)
    require_digest(model_sha256, "model_sha256")
    require_int(max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS)
    require_int(max_records, "max_records", 1, MAX_JOURNAL_RECORDS)
    return (
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP,
        str(Path(__file__).resolve().parent),
        "--source",
        str(source_path),
        "--journal",
        str(journal_path),
        "--lifetime-id",
        lifetime_id,
        "--model-sha256",
        model_sha256,
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-records",
        str(max_records),
    )


@dataclass(frozen=True)
class ServerHandshake:
    """Readiness facts a server lifetime publishes after binding its socket."""

    lifetime_id: str
    pid: int
    port: int
    model_sha256: str

    def __post_init__(self) -> None:
        require_lifetime_id(self.lifetime_id)
        require_int(self.pid, "pid", 1, 2**31 - 1)
        require_int(self.port, "port", 1, 65535)
        require_digest(self.model_sha256, "model_sha256")

    def to_line(self) -> bytes:
        return (
            canonical_json_bytes(
                {
                    "event": "ready",
                    "lifetime_id": self.lifetime_id,
                    "model_sha256": self.model_sha256,
                    "pid": self.pid,
                    "port": self.port,
                }
            )
            + b"\n"
        )


def parse_handshake(
    line: bytes, *, lifetime_id: str, pid: int, model_sha256: str
) -> ServerHandshake:
    """Accept a handshake only from the expected child, lifetime and model."""

    if type(line) is not bytes or not line.endswith(b"\n"):
        raise ServingRecoveryError("handshake must be one newline-terminated line")
    if len(line) > MAX_HANDSHAKE_BYTES:
        raise ServingRecoveryError("handshake exceeds its byte bound")
    payload = strict_json_loads(line[:-1], name="handshake")
    if type(payload) is not dict or set(payload) != {
        "event",
        "lifetime_id",
        "model_sha256",
        "pid",
        "port",
    }:
        raise ServingRecoveryError("handshake fields do not match schema")
    if payload["event"] != "ready":
        raise ServingRecoveryError("handshake does not declare readiness")
    handshake = ServerHandshake(
        payload["lifetime_id"], payload["pid"], payload["port"], payload["model_sha256"]
    )
    if handshake.to_line() != line:
        raise ServingRecoveryError("handshake is not canonical")
    if handshake.lifetime_id != lifetime_id:
        raise ServingRecoveryError("handshake names another server lifetime")
    if handshake.pid != pid:
        raise ServingRecoveryError("handshake pid is not the spawned child")
    if handshake.model_sha256 != model_sha256:
        raise ServingRecoveryError("handshake serves an unexpected model")
    return handshake


def health_body(*, lifetime_id: str, pid: int, model_sha256: str) -> bytes:
    """Return the exact ``GET /health`` body for one live lifetime."""

    return canonical_json_bytes(
        {
            "lifetime_id": lifetime_id,
            "model_sha256": model_sha256,
            "pid": pid,
            "status": "ok",
        }
    )


class RecoveryHTTPServer(ThreadingHTTPServer):
    """Loopback server with bounded connections and draining shutdown."""

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        *,
        model: ServingModel,
        journal: ExecutionJournal,
        lifetime_id: str,
        max_new_tokens: int,
    ) -> None:
        if journal.model_sha256 != model.model_sha256:
            raise ServingRecoveryError("journal and model identities differ")
        if journal.lifetime_id != lifetime_id:
            raise ServingRecoveryError("journal and server lifetimes differ")
        self.model = model
        self.journal = journal
        self.lifetime_id = lifetime_id
        self.max_new_tokens = require_int(
            max_new_tokens, "max_new_tokens", 1, MAX_NEW_TOKENS
        )
        self.health = health_body(
            lifetime_id=lifetime_id, pid=os.getpid(), model_sha256=model.model_sha256
        )
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__((LOOPBACK, 0), _GenerationHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind performs a reverse DNS lookup; skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = LOOPBACK, self.server_address[1]

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            body = canonical_json_bytes({"error": "connection limit reached"})
            try:
                request.settimeout(1)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        sys.stderr.write(f"request failed: {type(error).__name__}\n")


class _GenerationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "ai-journey-serving-recovery/1"
    sys_version = ""
    timeout = SOCKET_TIMEOUT_SECONDS
    server: RecoveryHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _reply(self, status: int, body: bytes, *, identified: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if identified:
            self.send_header(LIFETIME_HEADER, self.server.lifetime_id)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._reply(status, canonical_json_bytes({"error": message}))

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(404, "unknown route")
            return
        self._reply(200, self.server.health, identified=True)

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._error(404, "unknown route")
            return
        try:
            key, attempt, request = self._generation_request()
            body = completion_body(self.server.model, request)
        except (ServingRecoveryError, TypeError):
            self._error(400, "invalid generation request")
            return
        try:
            self.server.journal.append(
                idempotency_key=key,
                attempt=attempt,
                request_sha256=request.request_sha256(),
                response_sha256=sha256_hex(body),
            )
        except ServingRecoveryError:
            self._error(503, "execution journal is unavailable")
            return
        self._reply(200, body, identified=True)

    def _single_header(self, name: str) -> str:
        values = self.headers.get_all(name, [])
        if len(values) != 1:
            raise ServingRecoveryError(f"exactly one {name} header is required")
        return values[0]

    def _generation_request(self):
        length = self._single_header("Content-Length")
        if self.headers.get("Transfer-Encoding") is not None:
            raise ServingRecoveryError("chunked request bodies are unsupported")
        if _DECIMAL.fullmatch(length) is None or int(length) > MAX_BODY_BYTES:
            raise ServingRecoveryError("Content-Length is outside its bound")
        # Consume the declared body first so a rejection is not lost to a reset.
        body = self.rfile.read(int(length))
        if len(body) != int(length):
            raise ServingRecoveryError("request body is truncated")
        key = require_idempotency_key(self._single_header(KEY_HEADER))
        attempt = self._single_header(ATTEMPT_HEADER)
        if _DECIMAL.fullmatch(attempt) is None:
            raise ServingRecoveryError("attempt must be a positive decimal")
        attempt_number = require_int(int(attempt), "attempt", 1, MAX_ATTEMPT)
        request = request_from_mapping(strict_json_loads(body, name="request"))
        if request.max_new_tokens > self.server.max_new_tokens:
            raise ServingRecoveryError("max_new_tokens exceeds the server budget")
        return key, attempt_number, request


def _publish_handshake(handshake: ServerHandshake) -> None:
    line = memoryview(handshake.to_line())
    while line:
        line = line[os.write(1, line) :]
    # Nothing else is written to the handshake pipe after readiness.
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
    finally:
        os.close(devnull)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the recovery lab model.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--lifetime-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one server lifetime until SIGTERM; return 2 on invalid inputs."""

    arguments = _parser().parse_args(argv)
    try:
        lifetime_id = require_lifetime_id(arguments.lifetime_id)
        expected = require_digest(arguments.model_sha256, "model_sha256")
        source = read_bounded_file(
            arguments.source, max_bytes=MAX_SOURCE_BYTES, name="source"
        )
        model = build_serving_model(source)
        if model.model_sha256 != expected:
            raise ServingRecoveryError("source does not build the expected model")
        journal = ExecutionJournal(
            arguments.journal,
            lifetime_id=lifetime_id,
            model_sha256=expected,
            max_records=arguments.max_records,
        )
    except (OSError, ServingRecoveryError, TypeError) as exc:
        sys.stderr.write(f"serving recovery server: {exc}\n")
        return 2
    try:
        server = RecoveryHTTPServer(
            model=model,
            journal=journal,
            lifetime_id=lifetime_id,
            max_new_tokens=arguments.max_new_tokens,
        )
    except (OSError, ServingRecoveryError, TypeError) as exc:
        journal.close()
        sys.stderr.write(f"serving recovery server: {exc}\n")
        return 2

    def stop(signum: int, frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        _publish_handshake(
            ServerHandshake(
                lifetime_id, os.getpid(), server.server_address[1], expected
            )
        )
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        journal.close()
    return 0
