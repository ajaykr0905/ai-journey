"""Bounded, reproducible full-transformer training on verified public text."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from .gpt_bigram import CorpusSource, validate_corpus_bytes
from .transformer_lab import CharacterCodec, TokenCorpus, TransformerLabError


@dataclass(frozen=True)
class Day38DataConfig:
    """Explicit limits for a scale-limited public-corpus experiment."""

    block_size: int = 32
    train_token_limit: int = 65_536
    validation_token_limit: int = 8_192
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name in ("block_size", "train_token_limit", "validation_token_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= self.block_size if name != "block_size" else value <= 0:
                requirement = (
                    "positive" if name == "block_size" else "greater than block_size"
                )
                raise TransformerLabError(f"{name} must be {requirement}")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise TransformerLabError(
                "validation_fraction must be finite and between zero and one"
            )


@dataclass(frozen=True)
class BoundedCorpus:
    """A disjoint ordered split with explicit full- and used-token counts."""

    corpus: TokenCorpus
    source: CorpusSource
    total_token_count: int
    split_index: int
    full_train_token_count: int
    full_validation_token_count: int

    @property
    def used_train_token_count(self) -> int:
        return len(self.corpus.train_tokens)

    @property
    def used_validation_token_count(self) -> int:
        return len(self.corpus.validation_tokens)


def prepare_bounded_corpus(
    text: str, source: CorpusSource, config: Day38DataConfig
) -> BoundedCorpus:
    """Verify source bytes and return deterministic, disjoint bounded partitions."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(source, CorpusSource):
        raise TypeError("source must be CorpusSource")
    if not isinstance(config, Day38DataConfig):
        raise TypeError("config must be Day38DataConfig")
    validate_corpus_bytes(text.encode("utf-8"), source)
    codec = CharacterCodec.from_text(text)
    tokens = torch.tensor(codec.encode(text), dtype=torch.long)
    split_index = int(len(tokens) * (1 - config.validation_fraction))
    full_train = tokens[:split_index]
    full_validation = tokens[split_index:]
    minimum = config.block_size + 1
    if len(full_train) < minimum or len(full_validation) < minimum:
        raise TransformerLabError(
            "source split must provide more than block_size tokens per partition"
        )
    train = full_train[: config.train_token_limit].clone()
    validation = full_validation[: config.validation_token_limit].clone()
    if len(train) < minimum or len(validation) < minimum:
        raise TransformerLabError(
            "bounded partitions must provide more than block_size tokens"
        )
    corpus = TokenCorpus(
        codec=codec,
        train_tokens=train,
        validation_tokens=validation,
        source_sha256=source.sha256,
    )
    return BoundedCorpus(
        corpus=corpus,
        source=source,
        total_token_count=len(tokens),
        split_index=split_index,
        full_train_token_count=len(full_train),
        full_validation_token_count=len(full_validation),
    )
