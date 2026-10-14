"""Read public CoNLL-U sentence text without treating annotation rows as prose."""

from dataclasses import dataclass
from hashlib import sha256

from .bpe import TokenizerError


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str

    def __post_init__(self):
        if (
            not isinstance(self.sentence_id, str)
            or not self.sentence_id
            or any(character.isspace() for character in self.sentence_id)
            or not isinstance(self.text, str)
            or not self.text.strip()
        ):
            raise TokenizerError("sentence requires a stable id and nonempty text")


def extract_sentences(payload: bytes) -> tuple[Sentence, ...]:
    """Require one sent_id and text per nonempty CoNLL-U record."""
    if not isinstance(payload, bytes):
        raise TypeError("corpus payload must be bytes")
    if len(payload) > 8 * 1024 * 1024:
        raise TokenizerError("annotated corpus exceeds 8 MiB")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise TokenizerError("annotated corpus must be UTF-8") from exc
    records = []
    metadata = {}
    active = False
    seen = set()
    for line in [*text.split("\n"), ""]:
        line = line.removesuffix("\r")
        if not line:
            if not active:
                continue
            if set(metadata) != {"sent_id", "text"}:
                raise TokenizerError("sentence requires sent_id and text metadata")
            sentence_id, sentence = metadata["sent_id"], metadata["text"]
            if (
                not sentence_id
                or any(character.isspace() for character in sentence_id)
                or sentence_id in seen
                or not sentence.strip()
            ):
                raise TokenizerError("sentence ids must be unique and texts nonempty")
            seen.add(sentence_id)
            records.append(Sentence(sentence_id, sentence))
            metadata = {}
            active = False
            continue
        active = True
        for name in ("sent_id", "text"):
            prefix = f"# {name} = "
            if line.startswith(prefix):
                if name in metadata:
                    raise TokenizerError("duplicate sentence metadata")
                metadata[name] = line[len(prefix) :]
    if not records:
        raise TokenizerError("corpus contains no complete sentences")
    return tuple(records)


@dataclass(frozen=True)
class ParallelSplit:
    first_train: tuple[Sentence, ...]
    second_train: tuple[Sentence, ...]
    first_validation: tuple[Sentence, ...]
    second_validation: tuple[Sentence, ...]


def align_and_split(
    first: tuple[Sentence, ...],
    second: tuple[Sentence, ...],
    *,
    train_count: int,
    seed: int = 40,
) -> ParallelSplit:
    """Assign the same translation ids to train/validation using a stable hash."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise TokenizerError("split seed must be in [0, 2**32)")
    mappings = []
    for records in (first, second):
        if not isinstance(records, tuple) or not all(
            isinstance(row, Sentence) for row in records
        ):
            raise TypeError("parallel corpora must be tuples of Sentence records")
        mapping = {row.sentence_id: row for row in records}
        if len(mapping) != len(records):
            raise TokenizerError("parallel ids must be unique")
        mappings.append(mapping)
    left, right = mappings
    if left.keys() != right.keys():
        raise TokenizerError("parallel corpora must contain the same sentence ids")
    if type(train_count) is not int or not 1 <= train_count < len(left):
        raise TokenizerError("split must leave nonempty train and held-out sets")
    ids = sorted(
        left, key=lambda value: (sha256(f"{seed}:{value}".encode()).digest(), value)
    )
    train, heldout = ids[:train_count], ids[train_count:]
    return ParallelSplit(
        tuple(left[value] for value in train),
        tuple(right[value] for value in train),
        tuple(left[value] for value in heldout),
        tuple(right[value] for value in heldout),
    )
