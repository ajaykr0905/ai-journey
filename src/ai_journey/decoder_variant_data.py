"""Content-disjoint document partitions and matched within-document batches."""

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
from itertools import accumulate
import json

import torch

from .ddp_evaluation import Document, documents_from_lines


@dataclass(frozen=True)
class DecoderDocuments:
    train: tuple[Document, ...]
    heldout: tuple[Document, ...]
    context_length: int

    def evidence(self) -> dict:
        """Publish identities and coverage without publishing source text."""

        return {
            "train_documents": [doc.digest for doc in self.train],
            "heldout_documents": [doc.digest for doc in self.heldout],
            "train_window_count": sum(
                max(0, len(doc.content) - self.context_length) for doc in self.train
            ),
            "heldout_target_tokens": sum(len(doc.content) - 1 for doc in self.heldout),
        }


def prepare_decoder_documents(
    train_bytes: bytes, heldout_bytes: bytes, *, context_length: int
) -> DecoderDocuments:
    """Treat each nonempty line as a document, rejecting all exact leakage."""

    if type(context_length) is not int or not 1 <= context_length <= 64:
        raise ValueError("document context_length must be an integer in [1, 64]")
    if any(not isinstance(value, bytes) for value in (train_bytes, heldout_bytes)):
        raise TypeError("document sources must be bytes")
    if any(len(value) > 1024 * 1024 for value in (train_bytes, heldout_bytes)):
        raise ValueError("document source exceeds one MiB")
    train, heldout = (
        documents_from_lines(train_bytes),
        documents_from_lines(heldout_bytes),
    )
    if len(train) + len(heldout) > 2048:
        raise ValueError("document partitions exceed 2048 documents")
    if {doc.digest for doc in train} & {doc.digest for doc in heldout}:
        raise ValueError("training and held-out documents must be content-disjoint")
    if not any(len(doc.content) > context_length for doc in train):
        raise ValueError("training documents contain no complete context window")
    return DecoderDocuments(train, heldout, context_length)


def document_training_schedule(
    documents: DecoderDocuments, *, seed: int, steps: int, batch_size: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Uniformly sample valid document-local windows without a giant index."""

    for name, value, minimum, maximum in (
        ("seed", seed, 0, 2**63 - 1),
        ("steps", steps, 1, 64),
        ("batch_size", batch_size, 1, 32),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid document schedule {name}")
    cumulative = list(
        accumulate(
            max(0, len(doc.content) - documents.context_length)
            for doc in documents.train
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    offsets = torch.randint(
        cumulative[-1], (steps, batch_size), generator=generator, device="cpu"
    ).tolist()
    batches = []
    for batch in offsets:
        rows = []
        for offset in batch:
            index = bisect_right(cumulative, offset)
            start = offset - (cumulative[index - 1] if index else 0)
            rows.append((index, start))
        batches.append(tuple(rows))
    return tuple(batches)


def document_training_batch(documents: DecoderDocuments, rows):
    """Admit complete bounded document-local windows before tensor allocation."""

    if not isinstance(documents, DecoderDocuments):
        raise ValueError("document training requires DecoderDocuments")
    if not isinstance(rows, (tuple, list)) or not 1 <= len(rows) <= 32:
        raise ValueError("document training rows must contain one to 32 windows")
    for row in rows:
        if (
            not isinstance(row, (tuple, list))
            or len(row) != 2
            or any(type(value) is not int for value in row)
        ):
            raise ValueError(
                "document training windows require integer index/start pairs"
            )
        index, start = row
        if not 0 <= index < len(documents.train):
            raise ValueError(
                "document training index is outside the training partition"
            )
        last_start = len(documents.train[index].content) - documents.context_length - 1
        if not 0 <= start <= last_start:
            raise ValueError("document training window is incomplete or out of bounds")
    pieces = [
        documents.train[index].content[start : start + documents.context_length + 1]
        for index, start in rows
    ]
    values = torch.tensor(
        [list(piece) for piece in pieces], dtype=torch.long, device="cpu"
    )
    return values[:, :-1], values[:, 1:]


def document_schedule_fingerprint(schedule) -> str:
    return sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
