"""Pinned public-source adapter for the Day 59 English PUD dataset."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .data_documents import DataDocument, documents_from_sentences
from .tokenizer_sources import ENGLISH, PublicCorpus, load_public_corpus


@dataclass(frozen=True)
class SourceEvidence:
    metadata: dict
    source_url: str
    license_url: str
    documents: tuple[DataDocument, ...]


def load_pipeline_source(
    snapshot: Path, source: PublicCorpus = ENGLISH
) -> SourceEvidence:
    """Validate immutable source bytes before exposing training documents."""
    if not isinstance(snapshot, Path) or not isinstance(source, PublicCorpus):
        raise TypeError("pipeline source requires a snapshot Path and PublicCorpus")
    records = load_public_corpus(snapshot, source)
    documents = documents_from_sentences(records, source=source.url)
    return SourceEvidence(asdict(source), source.url, source.license_url, documents)
