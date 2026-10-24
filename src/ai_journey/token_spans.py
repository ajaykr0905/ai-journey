"""Exact byte offsets and character coverage for byte-level tokens."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .bpe import ByteTokenizer, SpecialTokenizer, TokenizerError


@dataclass(frozen=True)
class TokenSpan:
    token_id: int
    byte_start: int
    byte_end: int
    character_start: int
    character_end: int


@dataclass(frozen=True)
class ByteTokenSpan:
    token_id: int
    byte_start: int
    byte_end: int


def encode_bytes_with_spans(
    tokenizer: ByteTokenizer, payload: bytes
) -> tuple[ByteTokenSpan, ...]:
    if type(tokenizer) is not ByteTokenizer:
        raise TypeError("byte spans require a plain byte tokenizer")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    offset = 0
    spans = []
    for token_id in tokenizer.encode_bytes(payload):
        token_bytes = tokenizer.token_bytes(token_id)
        end = offset + len(token_bytes)
        if payload[offset:end] != token_bytes:
            raise TokenizerError("token bytes do not align with the input payload")
        spans.append(ByteTokenSpan(token_id, offset, end))
        offset = end
    if offset != len(payload):
        raise TokenizerError("token spans do not cover the complete payload")
    return tuple(spans)


def encode_with_spans(
    tokenizer, text: str, *, allowed=frozenset()
) -> tuple[TokenSpan, ...]:
    """Partial UTF-8 tokens share the containing character's coverage span."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(tokenizer, SpecialTokenizer):
        values = tokenizer.encode(text, allowed=allowed)
    else:
        if allowed:
            raise TokenizerError("ordinary tokenizer cannot authorize special markers")
        values = tokenizer.encode(text)
    encoded = text.encode("utf-8")
    boundaries = [0]
    for character in text:
        boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
    offset = 0
    spans = []
    for value in values:
        raw = tokenizer.token_bytes(value)
        end = offset + len(raw)
        if encoded[offset:end] != raw:
            raise TokenizerError("token bytes do not align with the original text")
        spans.append(
            TokenSpan(
                value,
                offset,
                end,
                bisect_right(boundaries, offset) - 1,
                bisect_left(boundaries, end),
            )
        )
        offset = end
    if offset != len(encoded):
        raise TokenizerError("token spans do not cover the complete text")
    return tuple(spans)
