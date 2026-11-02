"""Stable balanced sharding for reproducible dataset artifacts."""

from dataclasses import dataclass
from hashlib import sha256

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument


@dataclass(frozen=True)
class DataShard:
    index: int
    documents: tuple[CleanedDocument, ...]
    utf8_bytes: int


def shard_documents(
    documents: tuple[CleanedDocument, ...], *, shard_count: int, seed: int = 59
) -> tuple[DataShard, ...]:
    """Hash-rank documents then assign round-robin to balanced stable shards."""
    if (
        not isinstance(documents, tuple)
        or not documents
        or not all(isinstance(document, CleanedDocument) for document in documents)
    ):
        raise TokenizerError("sharding requires nonempty cleaned documents")
    if type(shard_count) is not int or not 1 <= shard_count <= len(documents):
        raise TokenizerError("shard count must be in [1, document count]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TokenizerError("shard seed must be in [0, 2**32)")
    if len({document.document_id for document in documents}) != len(documents):
        raise TokenizerError("sharding requires unique document ids")
    ordered = sorted(
        documents,
        key=lambda row: (
            sha256(f"{seed}:{row.document_id}".encode()).digest(),
            row.document_id,
        ),
    )
    buckets = [[] for _ in range(shard_count)]
    for index, document in enumerate(ordered):
        buckets[index % shard_count].append(document)
    return tuple(
        DataShard(
            index,
            tuple(bucket),
            sum(len(document.text.encode("utf-8")) for document in bucket),
        )
        for index, bucket in enumerate(buckets)
    )
