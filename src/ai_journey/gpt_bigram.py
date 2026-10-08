"""Day 34: deterministic character bigram training on Tiny Shakespeare."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor


class GPTBigramError(ValueError):
    """Raised when Day 34 data, model, or evidence contracts are invalid."""


@dataclass(frozen=True)
class CorpusSource:
    """Immutable provenance for an exact public corpus payload."""

    name: str
    url: str
    sha256: str
    byte_count: int
    license: str

    def __post_init__(self) -> None:
        for field_name in ("name", "url", "license"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise GPTBigramError(f"{field_name} must be a non-empty string")
        if not self.url.startswith("https://"):
            raise GPTBigramError("url must use HTTPS")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise GPTBigramError("sha256 must be 64 lowercase hexadecimal characters")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int):
            raise TypeError("byte_count must be an integer")
        if self.byte_count <= 0:
            raise GPTBigramError("byte_count must be positive")


@dataclass(frozen=True)
class CharacterVocabulary:
    """Stable character-to-id mapping derived from one corpus."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise GPTBigramError("vocabulary requires at least two characters")
        if any(not isinstance(token, str) or len(token) != 1 for token in self.tokens):
            raise GPTBigramError("vocabulary tokens must be single characters")
        if self.tokens != tuple(sorted(set(self.tokens))):
            raise GPTBigramError("vocabulary tokens must be unique and sorted")

    @classmethod
    def from_text(cls, text: str) -> CharacterVocabulary:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        indexes = {token: index for index, token in enumerate(self.tokens)}
        try:
            return tuple(indexes[token] for token in text)
        except KeyError as exc:
            raise GPTBigramError(f"unknown character: {exc.args[0]!r}") from exc

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token ids must be integers")
            if not 0 <= token_id < len(self.tokens):
                raise GPTBigramError(f"token id out of range: {token_id}")
            decoded.append(self.tokens[token_id])
        return "".join(decoded)


@dataclass(frozen=True)
class CorpusSplit:
    """Ordered train and held-out token partitions."""

    train: Tensor
    validation: Tensor
    split_index: int


def tokenize_and_split(
    text: str,
    vocabulary: CharacterVocabulary,
    *,
    validation_fraction: float = 0.1,
) -> CorpusSplit:
    """Encode text and reserve its final contiguous portion for validation."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(vocabulary, CharacterVocabulary):
        raise TypeError("vocabulary must be CharacterVocabulary")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0 < validation_fraction < 1
    ):
        raise GPTBigramError("validation_fraction must be between zero and one")
    tokens = torch.tensor(vocabulary.encode(text), dtype=torch.long)
    split_index = int(len(tokens) * (1 - validation_fraction))
    if split_index < 2 or len(tokens) - split_index < 2:
        raise GPTBigramError("train and validation partitions need at least two tokens")
    return CorpusSplit(
        train=tokens[:split_index].clone(),
        validation=tokens[split_index:].clone(),
        split_index=split_index,
    )


TINY_SHAKESPEARE = CorpusSource(
    name="Tiny Shakespeare",
    url=(
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    ),
    sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    byte_count=1_115_394,
    license="Public-domain Shakespeare text; compilation distributed by Andrej Karpathy",
)


def validate_corpus_bytes(payload: bytes, source: CorpusSource) -> str:
    """Validate exact source bytes before decoding them as UTF-8 text."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(source, CorpusSource):
        raise TypeError("source must be CorpusSource")
    if len(payload) != source.byte_count:
        raise GPTBigramError(
            f"corpus byte count mismatch: expected {source.byte_count}, got {len(payload)}"
        )
    digest = sha256(payload).hexdigest()
    if digest != source.sha256:
        raise GPTBigramError(
            f"corpus sha256 mismatch: expected {source.sha256}, got {digest}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GPTBigramError("corpus must be valid UTF-8") from exc
    if len(text) < 2:
        raise GPTBigramError("corpus must contain at least two characters")
    return text


def load_verified_corpus(path: Path, source: CorpusSource) -> str:
    """Read a local corpus snapshot and enforce its pinned source contract."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GPTBigramError(f"unable to read corpus snapshot: {path}") from exc
    return validate_corpus_bytes(payload, source)
