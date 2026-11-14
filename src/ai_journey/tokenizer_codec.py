"""BPE codec identity for transformer datasets and exact checkpoint binding."""

from dataclasses import dataclass

from .bpe import ByteTokenizer, SpecialTokenizer
from .tokenizer_io import tokenizer_fingerprint


@dataclass(frozen=True)
class BPECodec:
    """Codec interface that lets transformer corpora train on BPE ids.

    Its fingerprint enters the corpus fingerprint, so a checkpoint cannot be
    restored against a different vocabulary of the same size.
    """

    tokenizer: ByteTokenizer | SpecialTokenizer

    def __post_init__(self):
        if not isinstance(self.tokenizer, (ByteTokenizer, SpecialTokenizer)):
            raise TypeError("BPECodec requires a byte tokenizer")

    @property
    def vocab_size(self):
        """Includes reserved special ids, so embeddings cover every emitted id."""
        return self.tokenizer.vocab_size

    def encode(self, text):
        """Encode corpus text with no special markers authorized."""
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def fingerprint(self):
        """Tokenizer identity that corpus fingerprints and checkpoints bind to."""
        return tokenizer_fingerprint(self.tokenizer)
