"""Validated document records for reproducible public-data pipelines."""

from dataclasses import dataclass

from .bpe import TokenizerError
from .parallel_corpus import Sentence


@dataclass(frozen=True)
class DataDocument:
    """One source-addressable document before cleaning or tokenization."""

    document_id: str
    text: str
    source: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.document_id, str)
            or not self.document_id
            or any(character.isspace() for character in self.document_id)
        ):
            raise TokenizerError("document id must be a nonempty token")
        if not isinstance(self.text, str) or not self.text.strip():
            raise TokenizerError("document text must be nonempty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise TokenizerError("document source must be nonempty")


def documents_from_sentences(
    records: tuple[Sentence, ...], *, source: str
) -> tuple[DataDocument, ...]:
    """Preserve stable sentence identities while binding explicit provenance."""
    if (
        not isinstance(records, tuple)
        or not records
        or len(records) > 100_000
        or not all(isinstance(record, Sentence) for record in records)
    ):
        raise TokenizerError("documents require 1 to 100000 sentence records")
    documents = tuple(
        DataDocument(row.sentence_id, row.text, source) for row in records
    )
    if len({document.document_id for document in documents}) != len(documents):
        raise TokenizerError("document ids must be unique")
    return documents
