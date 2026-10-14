"""Train-only BPE fitting and explicit sentence boundaries for decoder training."""

from hashlib import sha256
import json

import torch

from .bpe import SpecialTokenizer, TokenizerError
from .tokenizer_codec import BPECodec
from .tokenizer_training import train_documents, training_segments
from .transformer_lab import TokenCorpus


EOS_MARKER = "<|endoftext|>"


def build_bpe_corpus(
    train: tuple[str, ...], validation: tuple[str, ...], *, vocab_size=288, pattern=None
):
    """Fit vocabulary on train only; append one reserved EOS per document."""
    training_segments(validation)
    tokenizer = SpecialTokenizer(
        train_documents(train, vocab_size=vocab_size, pattern=pattern), (EOS_MARKER,)
    )
    codec = BPECodec(tokenizer)
    eos = tokenizer.vocab_size - 1

    def encode(documents):
        # Ordinary documents cannot inject the registered boundary marker.
        return torch.tensor(
            [token for text in documents for token in (*codec.encode(text), eos)],
            dtype=torch.long,
        )

    if set(train) & set(validation):
        raise TokenizerError("exact training documents must not leak into validation")
    source = sha256(
        json.dumps(
            {"train": train, "validation": validation},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return TokenCorpus(codec, encode(train), encode(validation), source)
