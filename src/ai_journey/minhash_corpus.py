"""Verified Project 1 shard loading for MinHash deduplication."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument
from .data_pipeline_verify import verify_pipeline_artifacts


@dataclass(frozen=True)
class Project1Corpus:
    """Verified Day 59 documents with the report and shard digests evidence cites.

    Shards are re-hashed after verification, so MinHash evidence names exactly
    the bytes it deduplicated.
    """

    documents: tuple[CleanedDocument, ...]
    dataset_report_sha256: str
    shard_sha256s: tuple[str, ...]


def load_project1_corpus(directory: Path) -> Project1Corpus:
    """Load all verified train and validation shards with their identities."""
    if not isinstance(directory, Path):
        raise TypeError("Project 1 corpus root must be pathlib.Path")
    report = verify_pipeline_artifacts(directory)
    documents = []
    shard_sha256s = []
    for partition in ("train", "validation"):
        for descriptor in report["partitions"][partition]["shards"]:
            path = directory / descriptor["path"]
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise TokenizerError(
                    "unable to reread verified Project 1 shard"
                ) from exc
            if sha256(payload).hexdigest() != descriptor["sha256"]:
                raise TokenizerError("Project 1 shard changed after verification")
            shard_sha256s.append(descriptor["sha256"])
            for line in payload.splitlines():
                row = json.loads(line)
                documents.append(
                    CleanedDocument(
                        row["document_id"],
                        row["text"],
                        row["source"],
                        row["source_sha256"],
                        False,
                    )
                )
    ordered = tuple(sorted(documents, key=lambda row: row.document_id))
    if len({row.document_id for row in ordered}) != len(ordered):
        raise TokenizerError("Project 1 corpus document ids must be unique")
    return Project1Corpus(
        ordered,
        report["report_sha256"],
        tuple(shard_sha256s),
    )
