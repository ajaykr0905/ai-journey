"""Actual loopback HTTP observations; timings are client-observed receipt times."""

import http.client
import json
import math
import re
import socket
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .http_stream import (
    GenerationEventDecoder,
    StreamProtocolError,
    strict_json,
    validate_generation_output,
)


def loopback_address(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be a loopback HTTP URL")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
    ):
        raise ValueError(
            "endpoint must be http://127.0.0.1:PORT without credentials or a path"
        )
    return parsed.hostname, parsed.port


def validated_request(request):
    if type(request) is not dict or set(request) != {
        "request_id",
        "prompt",
        "sampling",
    }:
        raise ValueError("request must contain request_id, prompt and sampling")
    if (
        not isinstance(request["request_id"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["request_id"]) is None
    ):
        raise ValueError("invalid benchmark request_id")
    if (
        type(request["prompt"]) is not list
        or not 1 <= len(request["prompt"]) <= 512
        or any(
            type(token) is not int or not 0 <= token < 2**31
            for token in request["prompt"]
        )
    ):
        raise ValueError("prompt requires bounded nonnegative integer token ids")
    sampling = request["sampling"]
    defaults = {
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "seed": 48,
    }
    if type(sampling) is not dict or not set(sampling) <= set(defaults):
        raise ValueError("unsupported benchmark sampling controls")
    controls = defaults | sampling
    for name, minimum, maximum in (("max_new_tokens", 1, 128), ("seed", 0, 2**64 - 1)):
        if type(controls[name]) is not int or not minimum <= controls[name] <= maximum:
            raise ValueError(f"invalid benchmark {name}")
    if controls["top_k"] is not None and (
        type(controls["top_k"]) is not int or not 1 <= controls["top_k"] < 2**31
    ):
        raise ValueError("invalid benchmark top_k")
    for name in ("temperature", "top_p"):
        if (
            isinstance(controls[name], bool)
            or not isinstance(controls[name], (int, float))
            or not math.isfinite(controls[name])
        ):
            raise ValueError(f"invalid benchmark {name}")
    if controls["temperature"] < 0 or not 0 < controls["top_p"] <= 1:
        raise ValueError("invalid benchmark sampling range")
    return {
        "request_id": request["request_id"],
        "prompt": list(request["prompt"]),
        "sampling": controls,
    }


def measure_http_generation(
    endpoint, request, *, streaming=True, timeout_seconds=10.0, cancel_after_tokens=None
):
    """Measure one real connection, including connection setup and queueing.

    Token timestamps are captured when a socket read containing complete events
    returns. Coalesced events share a timestamp; this is not server compute time.
    JSON responses have no token-arrival timing. No retry conceals failed work.
    """

    host, port = loopback_address(endpoint)
    request = validated_request(request)
    if (
        type(streaming) is not bool
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 30
    ):
        raise ValueError("HTTP measurement timeout must be finite and in (0, 30]")
    maximum = request["sampling"]["max_new_tokens"]
    if cancel_after_tokens is not None and (
        not streaming
        or type(cancel_after_tokens) is not int
        or not 1 <= cancel_after_tokens < maximum
    ):
        raise ValueError("stream cancellation must precede its output token budget")
    body = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > 16 * 1024:
        raise ValueError("request exceeds the service body budget")
    started = time.perf_counter_ns()
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    response = None
    deadline_elapsed = threading.Event()
    transport_socket = None

    def expire_request():
        deadline_elapsed.set()
        if transport_socket is not None:
            try:
                transport_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    watchdog = threading.Timer(timeout_seconds, expire_request)
    watchdog.daemon = True
    result = {
        "request_id": request["request_id"],
        "route": "/stream" if streaming else "/generate",
        "started_ns": started,
        "headers_ns": None,
        "finished_ns": None,
        "status": None,
        "outcome": "transport_error",
        "tokens": [],
        "token_received_ns": [],
        "model_sha256": None,
        "finish_reason": None,
        "error_kind": None,
    }
    decoder = GenerationEventDecoder(
        request["request_id"], prompt_tokens=len(request["prompt"]), max_tokens=maximum
    )
    watchdog.start()
    try:
        # Capture the socket before getresponse(): HTTP/1.0 responses detach
        # it from HTTPConnection while the response's file object still reads.
        connection.connect()
        transport_socket = connection.sock
        if deadline_elapsed.is_set():
            raise TimeoutError("client connection deadline elapsed")
        connection.request(
            "POST",
            result["route"],
            body=body,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        result["headers_ns"] = time.perf_counter_ns()
        result["status"] = response.status
        if response.status != 200:
            response.read(4097)
            result["outcome"] = (
                "rejected" if response.status in {429, 503} else "http_error"
            )
            return result
        expected_type = "text/event-stream" if streaming else "application/json"
        if (
            response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            != expected_type
        ):
            raise StreamProtocolError("response content type does not match its route")
        if streaming:
            while True:
                if time.perf_counter_ns() - started >= timeout_seconds * 1e9:
                    raise TimeoutError("client receive deadline elapsed")
                chunk = response.read1(4096)
                received = time.perf_counter_ns()
                if not chunk:
                    break
                for event, payload in decoder.iter_events(chunk):
                    if event == "token":
                        result["tokens"].append(payload["token"])
                        result["token_received_ns"].append(received)
                        if cancel_after_tokens == len(result["tokens"]):
                            result["outcome"] = "cancelled"
                            result["error_kind"] = "client_closed_stream"
                            return result
            output = decoder.finish()
            if decoder.error is not None:
                result["outcome"] = "protocol_error"
                result["error_kind"] = "server_error_event"
                return result
        else:
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise StreamProtocolError("JSON response exceeds its byte budget")
            output = validate_generation_output(
                strict_json(payload),
                request_id=request["request_id"],
                prompt_tokens=len(request["prompt"]),
                max_tokens=maximum,
            )
            result["tokens"] = output["tokens"]
        result["model_sha256"] = output["model_sha256"]
        result["finish_reason"] = output["finish_reason"]
        result["outcome"] = (
            output["finish_reason"]
            if output["finish_reason"] in {"cancelled", "deadline"}
            else "completed"
        )
    except OSError as exc:
        result["outcome"] = (
            "deadline" if isinstance(exc, TimeoutError) else "transport_error"
        )
        result["error_kind"] = type(exc).__name__
    except (StreamProtocolError, ValueError, TypeError) as exc:
        result["outcome"] = "protocol_error"
        result["error_kind"] = type(exc).__name__
    except http.client.HTTPException as exc:
        result["outcome"] = "transport_error"
        result["error_kind"] = type(exc).__name__
    finally:
        watchdog.cancel()
        if response is not None:
            response.close()
        connection.close()
        result["finished_ns"] = time.perf_counter_ns()
        if (
            deadline_elapsed.is_set()
            or result["finished_ns"] - started >= timeout_seconds * 1e9
        ):
            result["outcome"] = "deadline"
            result["error_kind"] = "client_request_deadline"
    return result


def measure_http_output_parity(endpoint, request, *, timeout_seconds=10.0):
    """Use distinct fresh IDs so parity cannot be explained by response-cache hits."""

    request = validated_request(request)
    prefix = "parity-" + uuid4().hex
    stream = measure_http_generation(
        endpoint,
        request | {"request_id": prefix + "-s"},
        streaming=True,
        timeout_seconds=timeout_seconds,
    )
    plain = measure_http_generation(
        endpoint,
        request | {"request_id": prefix + "-j"},
        streaming=False,
        timeout_seconds=timeout_seconds,
    )
    matches = stream["outcome"] == plain["outcome"] == "completed" and all(
        stream[name] == plain[name]
        for name in ("tokens", "model_sha256", "finish_reason")
    )
    return {"stream": stream, "json": plain, "outputs_match": matches}
