"""Deterministic Unicode and whitespace cleaning with auditable source hashes."""

from dataclasses import dataclass
from hashlib import sha256
import unicodedata

from .bpe import TokenizerError
from .data_documents import DataDocument


@dataclass(frozen=True)
class CleanedDocument:
    """A normalized document that keeps the SHA-256 of its original text.

    changed records whether normalization rewrote it, so reports count edits
    instead of hiding them.
    """

    document_id: str
    text: str
    source: str
    source_sha256: str
    changed: bool


def clean_document(document: DataDocument) -> CleanedDocument:
    """Normalize NFC and whitespace without silently accepting control bytes."""
    if not isinstance(document, DataDocument):
        raise TypeError("cleaning requires a DataDocument")
    if "\ufffd" in document.text or any(
        unicodedata.category(character) == "Cc" and not character.isspace()
        for character in document.text
    ):
        raise TokenizerError("document contains replacement or control characters")
    normalized = " ".join(unicodedata.normalize("NFC", document.text).split())
    if not normalized:
        raise TokenizerError("cleaning produced an empty document")
    return CleanedDocument(
        document.document_id,
        normalized,
        document.source,
        sha256(document.text.encode("utf-8")).hexdigest(),
        normalized != document.text,
    )


def clean_documents(
    documents: tuple[DataDocument, ...],
) -> tuple[CleanedDocument, ...]:
    """Clean every document or fail the batch, so no input vanishes silently."""
    if not isinstance(documents, tuple) or not documents:
        raise TokenizerError("cleaning requires nonempty document input")
    return tuple(clean_document(document) for document in documents)
