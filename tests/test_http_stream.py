import json

import pytest

from ai_journey.http_stream import GenerationEventDecoder, StreamProtocolError


def frame(event, payload):
    return (
        f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
    )


def output(tokens=(3, 4)):
    return {
        "request_id": "request",
        "model_sha256": "a" * 64,
        "prompt_tokens": 2,
        "tokens": list(tokens),
        "finish_reason": "length",
    }


def stream():
    return (
        frame("token", {"index": 0, "token": 3})
        + frame("token", {"index": 1, "token": 4})
        + frame("done", output())
    )


def test_incremental_decoder_handles_every_byte_boundary_and_crlf():
    for payload in (stream(), stream().replace(b"\n", b"\r\n")):
        for boundary in range(len(payload) + 1):
            decoder = GenerationEventDecoder("request", prompt_tokens=2, max_tokens=2)
            decoder.feed(payload[:boundary])
            decoder.feed(payload[boundary:])
            assert decoder.finish() == output()


def test_utf8_error_terminal_and_truncated_codepoints():
    decoder = GenerationEventDecoder("request", prompt_tokens=2)
    for byte in frame("error", {"error": "défaillance"}):
        decoder.feed(bytes([byte]))
    assert decoder.finish() is None
    assert decoder.error == "défaillance"
    decoder = GenerationEventDecoder("request", prompt_tokens=2)
    decoder.feed(b"\xc3")
    with pytest.raises(StreamProtocolError, match="UTF-8"):
        decoder.finish()


@pytest.mark.parametrize(
    "payload",
    [
        frame("token", {"index": 1, "token": 3}),
        frame("token", {"index": 0, "token": True}),
        stream() + frame("done", output()),
        frame("done", output()),
        b'event: token\ndata: {"index":0,"index":0,"token":3}\n\n',
        b"event: unknown\ndata: {}\n\n",
    ],
)
def test_decoder_rejects_order_duplicates_parity_and_terminal_violations(payload):
    decoder = GenerationEventDecoder("request", prompt_tokens=2, max_tokens=2)
    with pytest.raises(StreamProtocolError):
        decoder.feed(payload)


def test_decoder_bounds_unterminated_events_and_requires_terminal():
    decoder = GenerationEventDecoder("request", prompt_tokens=2, max_event_bytes=8)
    with pytest.raises(StreamProtocolError, match="budget"):
        decoder.feed(b"event: token")
    decoder = GenerationEventDecoder("request", prompt_tokens=2)
    decoder.feed(frame("token", {"index": 0, "token": 3}))
    with pytest.raises(StreamProtocolError, match="terminal"):
        decoder.finish()
