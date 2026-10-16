"""Small authored, public-safe documents for a controlled CPU LoRA experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

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
