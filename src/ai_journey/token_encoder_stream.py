"""Bounded chunk ingestion with explicit document-safe BPE flushes."""

import base64
import binascii

from .bpe import ByteTokenizer, TokenizerError
from .tokenizer_io import tokenizer_fingerprint


class DocumentEncodingStream:
    """Buffer arbitrary chunks and encode only at declared document boundaries."""

    def __init__(self, tokenizer: ByteTokenizer, *, max_document_bytes=1024 * 1024):
        if type(tokenizer) is not ByteTokenizer:
            raise TypeError("streaming encoding requires a plain byte tokenizer")
        if (
            type(max_document_bytes) is not int
            or not 1 <= max_document_bytes <= 16 * 1024 * 1024
        ):
            raise TokenizerError("document budget must be in [1, 16777216]")
        self.tokenizer = tokenizer
        self.identity = tokenizer_fingerprint(tokenizer)
        self.max_document_bytes = max_document_bytes
        self._buffer = b""
        self._closed = False

    def feed(
        self, payload: bytes, *, document_end=False, final=False
    ) -> tuple[int, ...]:
        """Buffer a chunk and encode the buffer only at document_end or final.

        Merges always see the whole document, so chunk boundaries never change
        ids. Passing max_document_bytes or feeding after final raises
        TokenizerError.
        """
        if not isinstance(payload, bytes):
            raise TypeError("stream payload must be bytes")
        if not isinstance(document_end, bool) or not isinstance(final, bool):
            raise TypeError("document_end and final must be boolean")
        if self._closed:
            raise TokenizerError("token encoding stream is finalized")
        if len(self._buffer) + len(payload) > self.max_document_bytes:
            raise TokenizerError("token encoding stream exceeds its document budget")
        self._buffer += payload
        if not document_end and not final:
            return ()
        tokens = self.tokenizer.encode_bytes(self._buffer)
        self._buffer = b""
        self._closed = final
        return tokens

    def snapshot(self) -> dict:
        """Checkpoint the unflushed bytes with the tokenizer identity and budget."""
        return {
            "schema_version": 1,
            "tokenizer_sha256": self.identity,
            "max_document_bytes": self.max_document_bytes,
            "buffer": base64.b64encode(self._buffer).decode("ascii"),
            "finalized": self._closed,
        }

    def restore(self, state: dict) -> None:
        """Resume from a snapshot taken with the same tokenizer and budget.

        Other identities or budgets, malformed base64, an over-budget buffer or
        bytes left in a finalized stream raise TokenizerError and leave the live
        state unchanged.
        """
        if not isinstance(state, dict) or set(state) != {
            "schema_version",
            "tokenizer_sha256",
            "max_document_bytes",
            "buffer",
            "finalized",
        }:
            raise TokenizerError("token encoding snapshot fields are invalid")
        if (
            type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["tokenizer_sha256"] != self.identity
            or state["max_document_bytes"] != self.max_document_bytes
            or type(state["finalized"]) is not bool
            or not isinstance(state["buffer"], str)
        ):
            raise TokenizerError("token encoding snapshot policy is invalid")
        try:
            buffer = base64.b64decode(state["buffer"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise TokenizerError("token encoding snapshot buffer is invalid") from exc
        if len(buffer) > self.max_document_bytes or (state["finalized"] and buffer):
            raise TokenizerError("token encoding snapshot buffer exceeds its policy")
        self._buffer = buffer
        self._closed = state["finalized"]
