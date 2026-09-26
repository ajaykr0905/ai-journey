"""Deterministic model-selection utilities for the context language model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ai_journey.bigram_lm import normalize_corpus
from ai_journey.context_mlp import ContextMLPError


@dataclass(frozen=True)
class CorpusPartitions:
    """Whole-record train, development, and test partitions."""

    train: tuple[str, ...]
    development: tuple[str, ...]
    test: tuple[str, ...]


def split_train_dev_test(
    words: Iterable[str],
    *,
    development_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> CorpusPartitions:
    """Partition whole records deterministically without sample leakage."""

    corpus = normalize_corpus(words)
    if len(corpus) < 3:
        raise ContextMLPError("at least three records are required")
    for name, fraction in (
        ("development_fraction", development_fraction),
        ("test_fraction", test_fraction),
    ):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1
        ):
            raise ContextMLPError(f"{name} must be between zero and one")
    if development_fraction + test_fraction >= 1:
        raise ContextMLPError("development and test fractions must sum below one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    record_count = len(corpus)
    development_count = max(1, round(record_count * development_fraction))
    test_count = max(1, round(record_count * test_fraction))
    if development_count + test_count >= record_count:
        raise ContextMLPError("split fractions leave no training records")

    order = np.random.default_rng(seed).permutation(record_count)
    development_indexes = set(order[:development_count])
    test_indexes = set(order[development_count : development_count + test_count])
    return CorpusPartitions(
        train=tuple(
            word
            for index, word in enumerate(corpus)
            if index not in development_indexes | test_indexes
        ),
        development=tuple(
            word for index, word in enumerate(corpus) if index in development_indexes
        ),
        test=tuple(word for index, word in enumerate(corpus) if index in test_indexes),
    )
