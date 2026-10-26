"""Loopback-only HTTP transport for the bounded CPU inference lab."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time

from .ddp_contracts import integer
from .serving_metrics import PROMETHEUS_CONTENT_TYPE
from .trace_context import child_context, parse_traceparent
from .ddp_service import (
    AdmissionRejected,
    GenerationRequest,
    ServiceDraining,
    ServiceOverloaded,
    event_frame,
)


def _unique_fields(pairs):
    fields = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate JSON field")
        fields[name] = value
    return fields


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound accepted connections before creating per-connection worker threads."""

    def __init__(self, address, handler, *, max_connections):
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            body = b'{"error":"connection limit reached"}'
            response = (
                b"HTTP/1.0 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                b"Retry-After: 1\r\nConnection: close\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                + body
            )
            try:
                request.settimeout(1)
                request.sendall(response)
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

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(service, *, host="127.0.0.1", port=0, max_connections=8):
    if host != "127.0.0.1":
        raise ValueError("the unauthenticated lab server binds only to loopback")
    integer(max_connections, "max_connections", 1)
    if max_connections > 64:
        raise ValueError("max_connections exceeds the lab limit of 64")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):
            pass

        def start_trace(self):
            values = self.headers.get_all("traceparent", [])
            parent = parse_traceparent(values[0]) if len(values) == 1 else None
            self._trace_context = child_context(parent)

        def end_headers(self):
            context = getattr(self, "_trace_context", None)
            if context is not None:
                self.send_header("traceparent", context.to_header())
            super().end_headers()

        def reply(self, status, payload, *, retry_after=None):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if retry_after is not None:
                self.send_header("Retry-After", str(math.ceil(retry_after)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.start_trace()
            if self.path == "/health":
                self.reply(200, service.health())
            elif self.path == "/ready":
                health = service.health()
                self.reply(200 if health["ready"] else 503, health)
            elif self.path == "/metrics":
                body = service.metrics.render_prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", PROMETHEUS_CONTENT_TYPE)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.reply(404, {"error": "unknown route"})

        def do_POST(self):
            self.start_trace()
            if self.path not in ("/generate", "/stream"):
                self.reply(404, {"error": "unknown route"})
                return
            streaming = False

            def begin_stream():
                nonlocal streaming
                if not streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    streaming = True

            def emit(index, token):
                begin_stream()
                self.wfile.write(event_frame("token", {"index": index, "token": token}))
                self.wfile.flush()

            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdecimal()
                    or not 0 < int(lengths[0]) <= 16 * 1024
                    or self.headers.get("Transfer-Encoding")
                ):
                    raise ValueError("a bounded Content-Length is required")
                body = self.rfile.read(int(lengths[0]))
                if len(body) != int(lengths[0]):
                    raise ValueError("request body is truncated")
                try:
                    payload = json.loads(
                        body,
                        object_pairs_hook=_unique_fields,
                        parse_constant=_reject_constant,
                    )
                except RecursionError as error:
                    raise ValueError("JSON nesting exceeds parser limits") from error
                request = GenerationRequest.from_mapping(payload)
                output = service.generate(
                    request,
                    deadline=time.monotonic() + 10,
                    on_token=emit if self.path == "/stream" else None,
                )
                if self.path == "/stream":
                    begin_stream()
                    self.wfile.write(event_frame("done", output))
                else:
                    self.reply(200, output)
            except AdmissionRejected as error:
                self.reply(429, {"error": str(error)}, retry_after=error.retry_after)
            except ServiceDraining:
                self.reply(503, {"error": "service is draining"}, retry_after=1)
            except ServiceOverloaded:
                self.reply(503, {"error": "inference queue is full"}, retry_after=1)
            except (ValueError, TypeError, UnicodeError):
                if streaming:
                    self.wfile.write(
                        event_frame("error", {"error": "generation failed"})
                    )
                else:
                    self.reply(400, {"error": "invalid generation request"})
            except (RuntimeError, FloatingPointError):
                if streaming:
                    self.wfile.write(
                        event_frame("error", {"error": "generation failed"})
                    )
                else:
                    self.reply(500, {"error": "generation failed"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    return BoundedHTTPServer((host, port), Handler, max_connections=max_connections)
