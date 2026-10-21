"""Document-isolated byte-level evaluation for the distributed GPT-2 lab."""

from dataclasses import dataclass
from hashlib import sha256

from .ddp_contracts import integer


@dataclass(frozen=True)
class Document:
    identifier: str
    content: bytes

    def __post_init__(self):
        if (
            not isinstance(self.identifier, str)
            or not self.identifier.strip()
            or len(self.identifier) > 200
        ):
            raise ValueError("document identifier must be bounded nonempty text")
        if not isinstance(self.content, bytes) or len(self.content) < 2:
            raise ValueError("document content must contain at least two bytes")

    @property
    def digest(self) -> str:
        return sha256(self.content).hexdigest()


def documents_from_lines(source: bytes) -> tuple[Document, ...]:
    """Treat every nonempty public line as one document, rejecting duplicates."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    documents = tuple(
        Document(f"line-{number}", line + b"\n")
        for number, line in enumerate(source.splitlines(), 1)
        if line
    )
    if not documents or len({document.digest for document in documents}) != len(
        documents
    ):
        raise ValueError("documents must be nonempty and content-distinct")
    return documents


def split_documents(documents, *, validation_count: int, seed: int = 48):
    """Assign whole documents using keyed hashes, independent of input order."""
    documents = tuple(documents)
    integer(validation_count, "validation_count", 1)
    integer(seed, "seed")
    if validation_count >= len(documents):
        raise ValueError("training and validation must both contain documents")
    if len({doc.digest for doc in documents}) != len(documents) or len(
        {doc.identifier for doc in documents}
    ) != len(documents):
        raise ValueError("document identities and contents must be distinct")
    order = sorted(
        documents, key=lambda doc: sha256(f"{seed}:{doc.digest}".encode()).digest()
    )
    return tuple(order[validation_count:]), tuple(order[:validation_count])
