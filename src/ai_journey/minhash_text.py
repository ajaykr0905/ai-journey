"""Deterministic word shingles for approximate corpus deduplication."""

import re
import unicodedata

from .bpe import TokenizerError


_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_for_minhash(text: str) -> tuple[str, ...]:
    """Return case-folded Unicode word tokens with stable boundaries."""
    if not isinstance(text, str) or not text.strip():
        raise TokenizerError("MinHash text must be nonempty")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = tuple(_TOKEN.findall(normalized))
    if not tokens:
        raise TokenizerError("MinHash text must contain word tokens")
    return tokens


def word_shingles(text: str, *, width: int) -> tuple[str, ...]:
    """Build unique sorted word shingles, retaining one short-document shingle."""
    if type(width) is not int or not 1 <= width <= 32:
        raise TokenizerError("MinHash shingle width must be an integer from 1 to 32")
    tokens = tokenize_for_minhash(text)
    effective_width = min(width, len(tokens))
    shingles = {
        "\x1f".join(tokens[index : index + effective_width])
        for index in range(len(tokens) - effective_width + 1)
    }
    return tuple(sorted(shingles))
