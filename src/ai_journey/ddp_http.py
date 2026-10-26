"""Loopback-only HTTP transport for the bounded CPU inference lab."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import time

from .ddp_service import AdmissionRejected, GenerationRequest, event_frame


def make_server(service, *, host="127.0.0.1", port=0):
    if host != "127.0.0.1":
        raise ValueError("the unauthenticated lab server binds only to loopback")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):
            pass

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
            if self.path == "/health":
                self.reply(200, service.health())
            else:
                self.reply(404, {"error": "unknown route"})

        def do_POST(self):
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
                request = GenerationRequest.from_mapping(json.loads(body))
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

    return HTTPServer((host, port), Handler)
