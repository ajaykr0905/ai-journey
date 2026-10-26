"""Bounded incremental decoding of the CPU generation service's SSE protocol."""

import codecs
import json
import re


class StreamProtocolError(ValueError):
    """The received stream cannot constitute a complete generation response."""


def strict_json(payload):
    def unique(pairs):
        output = {}
        for name, value in pairs:
            if name in output:
                raise StreamProtocolError("duplicate JSON field")
            output[name] = value
        return output

    def constant(value):
        raise StreamProtocolError("non-finite JSON constant")

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StreamProtocolError("invalid bounded protocol JSON") from exc


def validate_generation_output(payload, *, request_id, prompt_tokens, max_tokens):
    fields = {"request_id", "model_sha256", "prompt_tokens", "tokens", "finish_reason"}
    if type(payload) is not dict or set(payload) != fields:
        raise StreamProtocolError("generation output fields differ from the protocol")
    if (
        payload["request_id"] != request_id
        or type(payload["prompt_tokens"]) is not int
        or payload["prompt_tokens"] != prompt_tokens
    ):
        raise StreamProtocolError("generation output does not match the request")
    if (
        type(payload["model_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", payload["model_sha256"]) is None
    ):
        raise StreamProtocolError("generation output requires a model SHA-256")
    tokens = payload["tokens"]
    if (
        type(tokens) is not list
        or len(tokens) > max_tokens
        or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens)
    ):
        raise StreamProtocolError("generation output contains invalid token ids")
    if payload["finish_reason"] not in {"length", "eos", "cancelled", "deadline"}:
        raise StreamProtocolError("unsupported generation finish reason")
    if payload["finish_reason"] == "length" and len(tokens) != max_tokens:
        raise StreamProtocolError("length completion does not exhaust its token budget")
    return payload


class GenerationEventDecoder:
    """Decode arbitrary byte boundaries, enforcing token order and one terminal."""

    def __init__(
        self,
        request_id,
        *,
        prompt_tokens,
        max_tokens=128,
        max_bytes=1024 * 1024,
        max_event_bytes=16 * 1024,
    ):
        if (
            not isinstance(request_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id) is None
        ):
            raise ValueError("request_id must use 1 to 64 safe identifier characters")
        for name, value, lower, upper in (
            ("prompt_tokens", prompt_tokens, 1, 512),
            ("max_tokens", max_tokens, 1, 128),
            ("max_bytes", max_bytes, 1, 1024 * 1024),
            ("max_event_bytes", max_event_bytes, 1, 64 * 1024),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"invalid stream {name}")
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_bytes = max_bytes
        self.max_event_bytes = max_event_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._text = ""
        self._lines = []
        self._event_bytes = 0
        self._total_bytes = 0
        self._closed = False
        self.tokens = []
        self.output = None
        self.error = None

    def _consume(self, text):
        self._text += text
        events = []
        while "\n" in self._text:
            line, self._text = self._text.split("\n", 1)
            line = line.removesuffix("\r")
            if "\r" in line:
                raise StreamProtocolError("unsupported SSE line ending")
            self._event_bytes += len(line.encode()) + 1
            if self._event_bytes > self.max_event_bytes:
                raise StreamProtocolError("SSE event exceeds its byte budget")
            if line:
                if self.output is not None or self.error is not None:
                    raise StreamProtocolError("event appeared after the terminal event")
                self._lines.append(line)
            elif self._lines:
                if (
                    len(self._lines) != 2
                    or not self._lines[0].startswith("event: ")
                    or not self._lines[1].startswith("data: ")
                ):
                    raise StreamProtocolError(
                        "generation SSE requires one event and one data line"
                    )
                event = self._lines[0][7:]
                data = strict_json(self._lines[1][6:])
                self._lines = []
                self._event_bytes = 0
                if event == "token":
                    if (
                        type(data) is not dict
                        or set(data) != {"index", "token"}
                        or type(data["index"]) is not int
                        or data["index"] != len(self.tokens)
                        or type(data["token"]) is not int
                        or not 0 <= data["token"] < 2**31
                        or len(self.tokens) >= self.max_tokens
                    ):
                        raise StreamProtocolError(
                            "token event index, id or count is invalid"
                        )
                    self.tokens.append(data["token"])
                elif event == "done":
                    self.output = validate_generation_output(
                        data,
                        request_id=self.request_id,
                        prompt_tokens=self.prompt_tokens,
                        max_tokens=self.max_tokens,
                    )
                    if self.output["tokens"] != self.tokens:
                        raise StreamProtocolError(
                            "terminal tokens differ from streamed tokens"
                        )
                elif event == "error":
                    if (
                        type(data) is not dict
                        or set(data) != {"error"}
                        or type(data["error"]) is not str
                        or not 1 <= len(data["error"]) <= 256
                    ):
                        raise StreamProtocolError("invalid error terminal event")
                    self.error = data["error"]
                else:
                    raise StreamProtocolError("unsupported generation SSE event")
                events.append((event, data))
        if self._event_bytes + len(self._text.encode()) > self.max_event_bytes:
            raise StreamProtocolError("SSE event exceeds its byte budget")
        return events

    def feed(self, payload: bytes):
        if self._closed or not isinstance(payload, bytes):
            raise StreamProtocolError("stream decoder requires bytes before EOF")
        self._total_bytes += len(payload)
        if self._total_bytes > self.max_bytes:
            raise StreamProtocolError("SSE stream exceeds its byte budget")
        try:
            return self._consume(self._decoder.decode(payload))
        except UnicodeError as exc:
            raise StreamProtocolError("stream contains invalid UTF-8") from exc

    def finish(self):
        try:
            self._consume(self._decoder.decode(b"", final=True))
        except UnicodeError as exc:
            raise StreamProtocolError("stream ends inside a UTF-8 codepoint") from exc
        self._closed = True
        if self._text or self._lines or (self.output is None and self.error is None):
            raise StreamProtocolError("stream ended without a complete terminal event")
        return self.output
