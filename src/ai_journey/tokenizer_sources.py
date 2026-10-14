"""Pinned, licensed public tokenizer corpora; no network access during loading."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .bpe import TokenizerError
from .parallel_corpus import extract_sentences


@dataclass(frozen=True)
class PublicCorpus:
    language: str
    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    license: str = "CC-BY-SA-3.0"

    @property
    def url(self):
        return f"https://raw.githubusercontent.com/{self.repository}/{self.revision}/{self.filename}"

    @property
    def license_url(self):
        return f"https://github.com/{self.repository}/blob/{self.revision}/LICENSE.txt"


HINDI = PublicCorpus(
    "hi",
    "UniversalDependencies/UD_Hindi-PUD",
    "33e5ecd280d97c7b744993a1e12230559c94fc89",
    "hi_pud-ud-test.conllu",
    "cca405c5e008e1245fc4b4e89b5d8c0c04fc95e84341a7a9957bf277daae173d",
    3013600,
)
ENGLISH = PublicCorpus(
    "en",
    "UniversalDependencies/UD_English-PUD",
    "6e328e8195c04a297c1e375d21fe663be1ce6178",
    "en_pud-ud-test.conllu",
    "63eb7eca373bd45912c01f15f1e445d5cdd41245f20da1669474728e484e096d",
    1355100,
)


def load_public_corpus(path: Path, source: PublicCorpus):
    """Reject truncated, changed, or oversized snapshots before parsing text."""
    if not isinstance(path, Path) or not isinstance(source, PublicCorpus):
        raise TypeError("loading requires a Path and PublicCorpus")
    if (
        type(source.size_bytes) is not int
        or not 0 < source.size_bytes <= 8 * 1024 * 1024
    ):
        raise TokenizerError("source size must be positive and bounded")
    with path.open("rb") as stream:
        payload = stream.read(source.size_bytes + 1)
    if (
        len(payload) != source.size_bytes
        or sha256(payload).hexdigest() != source.sha256
    ):
        raise TokenizerError("public corpus snapshot identity mismatch")
    return extract_sentences(payload)
