"""Restartable UTF-8 decoding for token-at-a-time generation."""

import codecs
import base64
import binascii

from .bpe import TokenizerError
from .tokenizer_io import tokenizer_fingerprint


class TokenTextStream:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.identity = tokenizer_fingerprint(tokenizer)
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._closed = False

    def feed(self, values, *, final=False):
        if not isinstance(final, bool):
            raise TypeError("final must be boolean")
        if self._closed:
            raise TokenizerError("token text stream is finalized")
        if tokenizer_fingerprint(self.tokenizer) != self.identity:
            raise TokenizerError("tokenizer identity changed")
        payload = self.tokenizer.decode_bytes(values)
        previous = self._decoder.getstate()
        try:
            text = self._decoder.decode(payload, final=final)
        except UnicodeError:
            self._decoder.setstate(previous)
            raise
        self._closed = final
        return text

    def snapshot(self):
        buffered, _ = self._decoder.getstate()
        return {
            "schema_version": 1,
            "tokenizer_fingerprint": self.identity,
            "buffer": base64.b64encode(buffered).decode("ascii"),
            "finalized": self._closed,
        }

    def restore(self, state):
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "tokenizer_fingerprint",
            "buffer",
            "finalized",
        }:
            raise TokenizerError("token stream snapshot fields are invalid")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise TokenizerError("unsupported token stream version")
        if (
            state["tokenizer_fingerprint"] != self.identity
            or type(state["finalized"]) is not bool
        ):
            raise TokenizerError("token stream identity or finalized flag is invalid")
        if not isinstance(state["buffer"], str) or len(state["buffer"]) > 4:
            raise TokenizerError("invalid UTF-8 decoder buffer")
        try:
            raw = base64.b64decode(state["buffer"], validate=True)
            candidate = codecs.getincrementaldecoder("utf-8")("strict")
            emitted = candidate.decode(raw, final=False)
        except (UnicodeError, binascii.Error) as exc:
            raise TokenizerError("invalid UTF-8 decoder buffer") from exc
        if emitted or (state["finalized"] and raw):
            raise TokenizerError(
                "snapshot must contain only an unfinished UTF-8 prefix"
            )
        self._decoder = candidate
        self._closed = state["finalized"]
