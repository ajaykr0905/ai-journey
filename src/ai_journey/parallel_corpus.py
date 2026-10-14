"""Read public CoNLL-U sentence text without treating annotation rows as prose."""

from dataclasses import dataclass

from .bpe import TokenizerError


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str


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
