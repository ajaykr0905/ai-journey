"""Order-independent exact deduplication for admitted public documents."""

from dataclasses import dataclass
from hashlib import sha256

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument


@dataclass(frozen=True)
class DuplicateRecord:
    """One removed exact duplicate and the kept document with the same text."""

    duplicate_id: str
    canonical_id: str
    content_sha256: str


@dataclass(frozen=True)
class DeduplicationResult:
    """Kept documents in id order and a record for every removed duplicate."""

    documents: tuple[CleanedDocument, ...]
    duplicates: tuple[DuplicateRecord, ...]


def deduplicate_documents(
    documents: tuple[CleanedDocument, ...],
) -> DeduplicationResult:
    """Keep the lexicographically first id for each exact cleaned text."""
    if (
        not isinstance(documents, tuple)
        or not documents
        or not all(isinstance(document, CleanedDocument) for document in documents)
    ):
        raise TokenizerError("deduplication requires nonempty cleaned documents")
    ordered = sorted(documents, key=lambda row: (row.document_id, row.source))
    if len({row.document_id for row in ordered}) != len(ordered):
        raise TokenizerError("deduplication requires unique document ids")
    canonical = {}
    kept = []
    duplicates = []
    for document in ordered:
        digest = sha256(document.text.encode("utf-8")).hexdigest()
        match = canonical.get(digest)
        if match is None:
            canonical[digest] = document
            kept.append(document)
            continue
        if match.text != document.text:
            raise TokenizerError("content digest collision")
        duplicates.append(
            DuplicateRecord(document.document_id, match.document_id, digest)
        )
    return DeduplicationResult(tuple(kept), tuple(duplicates))
