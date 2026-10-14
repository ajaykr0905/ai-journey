"""Compression evidence with explicit byte, code-point, and token denominators."""

from collections import Counter
from dataclasses import asdict, dataclass
import math

from .bpe import ByteTokenizer, SpecialTokenizer, TokenizerError


@dataclass(frozen=True)
class CompressionMetrics:
    documents: int
    utf8_bytes: int
    characters: int
    tokens: int
    used_vocabulary: int
    byte_fallback_tokens: int
    bytes_per_token: float
    characters_per_token: float
    token_entropy_bits: float

    def as_dict(self):
        return asdict(self)


def compression_metrics(tokenizer, documents: tuple[str, ...]) -> CompressionMetrics:
    """Encode each document independently and verify every lossless round trip."""
    if not isinstance(tokenizer, (ByteTokenizer, SpecialTokenizer)):
        raise TypeError("expected a byte tokenizer")
    if (
        not isinstance(documents, tuple)
        or not documents
        or any(not isinstance(text, str) or not text for text in documents)
    ):
        raise TokenizerError("metrics require nonempty text documents")
    counts = Counter()
    byte_count = character_count = 0
    for text in documents:
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids) != text:
            raise TokenizerError("tokenizer failed lossless round-trip audit")
        counts.update(ids)
        byte_count += len(text.encode("utf-8"))
        character_count += len(text)
    total = sum(counts.values())
    entropy = -math.fsum(
        count / total * math.log2(count / total) for count in counts.values()
    )
    return CompressionMetrics(
        len(documents),
        byte_count,
        character_count,
        total,
        len(counts),
        sum(count for token, count in counts.items() if token < 256),
        byte_count / total,
        character_count / total,
        entropy,
    )
