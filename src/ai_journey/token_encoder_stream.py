"""Bounded chunk ingestion with explicit document-safe BPE flushes."""

from .bpe import ByteTokenizer, TokenizerError


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
        self.max_document_bytes = max_document_bytes
        self._buffer = b""
        self._closed = False

    def feed(
        self, payload: bytes, *, document_end=False, final=False
    ) -> tuple[int, ...]:
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
