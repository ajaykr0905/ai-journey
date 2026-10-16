"""Small authored, public-safe documents for a controlled CPU LoRA experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random

from .transformer_lab import CharacterCodec


class LoRAExperimentError(ValueError):
    """The bounded comparison's data, controls, or evidence are invalid."""


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str

    def __post_init__(self):
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise LoRAExperimentError("document_id must be a nonempty string")
        if not isinstance(self.text, str) or len(self.text) < 2:
            raise LoRAExperimentError(
                "document text must contain at least two characters"
            )


def canonical_digest(value) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class DocumentSplit:
    pretrain: tuple[Document, ...]
    train: tuple[Document, ...]
    validation: tuple[Document, ...]

    def __post_init__(self):
        partitions = (self.pretrain, self.train, self.validation)
        for partition in partitions:
            if (
                not isinstance(partition, tuple)
                or not partition
                or any(not isinstance(doc, Document) for doc in partition)
            ):
                raise LoRAExperimentError("partitions require nonempty document tuples")
        documents = sum(partitions, ())
        if len({doc.document_id for doc in documents}) != len(documents):
            raise LoRAExperimentError("document ids must be unique and disjoint")
        if len({doc.text for doc in documents}) != len(documents):
            raise LoRAExperimentError("duplicate text across documents is not allowed")
        # Fit only on documents that may participate in optimization.
        codec = self.codec()
        for doc in self.validation:
            try:
                codec.encode(doc.text)
            except ValueError as exc:
                raise LoRAExperimentError(
                    "validation contains an unseen character"
                ) from exc

    def codec(self) -> CharacterCodec:
        return CharacterCodec.from_text(
            "".join(doc.text for doc in self.pretrain + self.train)
        )

    def evidence(self) -> dict:
        result = {}
        for name in ("pretrain", "train", "validation"):
            documents = getattr(self, name)
            result[name] = {
                "document_ids": [doc.document_id for doc in documents],
                "sha256": canonical_digest(
                    [[doc.document_id, doc.text] for doc in documents]
                ),
                "characters": sum(len(doc.text) for doc in documents),
                "targets": sum(len(doc.text) - 1 for doc in documents),
            }
        result["vocabulary"] = list(self.codec().tokens)
        return result


FIXTURE_ID = "authored-tree-grammar-v1"


def synthetic_fixture() -> DocumentSplit:
    """Generate fixed artificial prose, without downloads or personal/source data.

    Partitions share an alphabet and grammar; entire document strings and ids are
    disjoint. This is a small transfer exercise, not natural-language evidence.
    """
    trees = ("cedar", "birch", "maple", "willow")
    states = ("calm", "warm", "cool", "mild")
    places = ("east", "west", "north", "south")
    pretrain, train, validation = [], [], []
    for index in range(8):
        pretrain.append(
            Document(
                f"pretrain-{index}",
                "".join(
                    f"{trees[j]} is {states[(j + index) % 4]} in {places[(j + index // 4) % 4]}.\n"
                    for j in range(4)
                ),
            )
        )
    for index in range(12):
        doc = Document(
            f"adapt-{index}" if index < 8 else f"heldout-{index - 8}",
            "".join(
                f"{states[(j + index) % 4]} {trees[j]} grows {places[(j + index // 4) % 4]}.\n"
                for j in range(4)
            ),
        )
        (train if index < 8 else validation).append(doc)
    return DocumentSplit(tuple(pretrain), tuple(train), tuple(validation))


def bounded_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise LoRAExperimentError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _encoded_documents(documents, codec):
    if (
        not isinstance(documents, tuple)
        or not documents
        or any(not isinstance(doc, Document) for doc in documents)
        or len({doc.document_id for doc in documents}) != len(documents)
    ):
        raise LoRAExperimentError("expected documents with unique ids")
    if not isinstance(codec, CharacterCodec):
        raise LoRAExperimentError("expected a training CharacterCodec")
    return tuple((doc.document_id, codec.encode(doc.text)) for doc in documents)


@dataclass(frozen=True)
class TrainingWindow:
    document_id: str
    start: int
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class HeldoutTarget:
    document_id: str
    target_index: int
    context: tuple[int, ...]
    target: int


def training_batches(documents, codec, *, block_size, batch_size, steps, seed):
    """Draw one immutable schedule with a private RNG; no document concatenation."""
    bounded_integer(block_size, "block_size", 1, 32)
    bounded_integer(batch_size, "batch_size", 1, 8)
    bounded_integer(steps, "steps", 1, 64)
    bounded_integer(seed, "seed", 0, 2**32 - 1)
    candidates = []
    for doc_id, tokens in _encoded_documents(documents, codec):
        if len(tokens) <= block_size:
            raise LoRAExperimentError(
                "every training document needs a full block and target"
            )
        for start in range(len(tokens) - block_size):
            candidates.append(
                TrainingWindow(
                    doc_id,
                    start,
                    tokens[start : start + block_size],
                    tokens[start + 1 : start + block_size + 1],
                )
            )
    rng = random.Random(seed)
    return tuple(
        tuple(rng.choice(candidates) for _ in range(batch_size)) for _ in range(steps)
    )


def schedule_evidence(batches):
    return {
        "steps": len(batches),
        "windows": sum(len(batch) for batch in batches),
        "target_tokens": sum(len(row.targets) for batch in batches for row in batch),
        "sha256": canonical_digest(
            [
                [[row.document_id, row.start, row.inputs, row.targets] for row in batch]
                for batch in batches
            ]
        ),
    }


def heldout_targets(documents, codec, *, block_size):
    """Each target 1..len(document)-1 is scored once with a bounded left context.

    The first character is context, never an unconditional target. Contexts may
    overlap, but targets do not; absolute positions restart within each window.
    """
    bounded_integer(block_size, "block_size", 1, 32)
    return tuple(
        HeldoutTarget(
            doc_id, index, tokens[max(0, index - block_size) : index], tokens[index]
        )
        for doc_id, tokens in _encoded_documents(documents, codec)
        for index in range(1, len(tokens))
    )


def heldout_evidence(documents, codec, *, block_size):
    targets = heldout_targets(documents, codec, block_size=block_size)
    return {
        "target_tokens": len(targets),
        "per_document": [
            {"document_id": doc.document_id, "target_tokens": len(doc.text) - 1}
            for doc in documents
        ],
        "sha256": canonical_digest(
            [
                [row.document_id, row.target_index, row.context, row.target]
                for row in targets
            ]
        ),
        "policy": "one-last-logit-per-target; document-local sliding context; first character excluded",
    }
