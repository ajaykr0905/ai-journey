"""Day 34: deterministic character bigram training on Tiny Shakespeare."""

from __future__ import annotations

from dataclasses import dataclass


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
