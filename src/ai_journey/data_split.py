"""Deterministic leakage-safe train and validation partitioning."""

from dataclasses import dataclass
from hashlib import sha256

from .bpe import TokenizerError
from .data_cleaning import CleanedDocument


@dataclass(frozen=True)
class DataSplit:
    """Disjoint train and validation partitions plus the seed that ranked them."""

    train: tuple[CleanedDocument, ...]
    validation: tuple[CleanedDocument, ...]
    seed: int


def split_documents(
    documents: tuple[CleanedDocument, ...], *, validation_count: int, seed: int = 59
) -> DataSplit:
    """Rank stable ids by a seeded digest and reserve an exact held-out count."""
    if (
        not isinstance(documents, tuple)
        or len(documents) < 2
        or not all(isinstance(document, CleanedDocument) for document in documents)
    ):
        raise TokenizerError("splitting requires at least two cleaned documents")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TokenizerError("split seed must be in [0, 2**32)")
    if type(validation_count) is not int or not 1 <= validation_count < len(documents):
        raise TokenizerError("validation count must leave nonempty partitions")
    identities = [document.document_id for document in documents]
    content = [sha256(document.text.encode("utf-8")).digest() for document in documents]
    if len(set(identities)) != len(identities) or len(set(content)) != len(content):
        raise TokenizerError("split input must have unique ids and cleaned content")
    ordered = sorted(
        documents,
        key=lambda row: (
            sha256(f"{seed}:{row.document_id}".encode()).digest(),
            row.document_id,
        ),
    )
    validation = tuple(ordered[:validation_count])
    train = tuple(ordered[validation_count:])
    return DataSplit(train, validation, seed)
