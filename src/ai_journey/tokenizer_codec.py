"""BPE codec identity for transformer datasets and exact checkpoint binding."""

from dataclasses import dataclass

from .bpe import ByteTokenizer, SpecialTokenizer
from .tokenizer_io import tokenizer_fingerprint


@dataclass(frozen=True)
class BPECodec:
    tokenizer: ByteTokenizer | SpecialTokenizer

    def __post_init__(self):
        if not isinstance(self.tokenizer, (ByteTokenizer, SpecialTokenizer)):
            raise TypeError("BPECodec requires a byte tokenizer")

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def fingerprint(self):
        return tokenizer_fingerprint(self.tokenizer)
