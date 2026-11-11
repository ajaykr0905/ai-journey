"""A fixed UTF-8 byte codec that never learns validation vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .sft_finetune_contract import SFTFineTuneError


@dataclass(frozen=True)
class UTF8ByteCodec:
    """Losslessly encode public text with a stable 256-token vocabulary."""

    vocab_size: int = 256

    def __post_init__(self) -> None:
        if self.vocab_size != 256:
            raise SFTFineTuneError("UTF-8 byte vocabulary must contain exactly 256 ids")

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        if any(type(value) is not int or not 0 <= value < 256 for value in token_ids):
            raise SFTFineTuneError("byte token ids must be integers in [0, 255]")
        try:
            return bytes(token_ids).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SFTFineTuneError("byte tokens are not complete UTF-8") from exc

    def fingerprint(self) -> str:
        return sha256(bytes(range(256))).hexdigest()
