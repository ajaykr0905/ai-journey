"""Pinned public-source adapter for the Day 59 English PUD dataset."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .data_documents import DataDocument, documents_from_sentences
from .tokenizer_sources import ENGLISH, PublicCorpus, load_public_corpus


PUD_ATTRIBUTION = (
    "Parallel Universal Dependencies, CoNLL 2017; DFKI, Google, and the UD "
    "community; see the source README for contributor credits."
)


@dataclass(frozen=True)
class SourceEvidence:
    """Validated documents plus the provenance the manifest must carry.

    metadata is the pinned PublicCorpus record; attribution must be nonempty
    because the CC-BY-SA source license requires credit.
    """

    metadata: dict
    source_url: str
    license_url: str
    documents: tuple[DataDocument, ...]
    attribution: str = PUD_ATTRIBUTION

    def __post_init__(self) -> None:
        if not isinstance(self.attribution, str) or not self.attribution.strip():
            raise ValueError("source attribution must be nonempty")


def load_pipeline_source(
    snapshot: Path, source: PublicCorpus = ENGLISH
) -> SourceEvidence:
    """Validate immutable source bytes before exposing training documents."""
    if not isinstance(snapshot, Path) or not isinstance(source, PublicCorpus):
        raise TypeError("pipeline source requires a snapshot Path and PublicCorpus")
    records = load_public_corpus(snapshot, source)
    documents = documents_from_sentences(records, source=source.url)
    return SourceEvidence(
        asdict(source), source.url, source.license_url, documents, PUD_ATTRIBUTION
    )
