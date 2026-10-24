"""Source-backed conformance checks for learned byte-pair merge tables."""

from collections import Counter
from dataclasses import dataclass

from .bpe import (
    ByteTokenizer,
    RegexTokenizer,
    TokenizerError,
    _merge_weighted_streams,
    _weighted_pair_counts,
)
from .tokenizer_training import training_segments


@dataclass(frozen=True)
class MergeObservation:
    """The canonical pair selected at one training rank."""

    rank: int
    pair: tuple[int, int]
    frequency: int
    applied_count: int
    tokens_before: int
    tokens_after: int


def _weighted_token_count(streams: Counter) -> int:
    return sum(len(stream) * weight for stream, weight in streams.items())


def replay_merge_prefix(
    documents: tuple[str, ...],
    tokenizer: ByteTokenizer,
    *,
    min_frequency: int = 2,
) -> tuple[MergeObservation, ...]:
    """Prove that every stored merge is the canonical next corpus merge."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("merge replay requires a byte tokenizer")
    ByteTokenizer.train("", min_frequency=min_frequency)
    pattern = tokenizer.pattern if isinstance(tokenizer, RegexTokenizer) else None
    streams = Counter(
        tuple(segment.encode("utf-8"))
        for segment in training_segments(documents, pattern=pattern)
    )
    observations = []
    for rank, stored_pair in enumerate(tokenizer.merges):
        counts = _weighted_pair_counts(streams)
        if not counts:
            raise TokenizerError("stored merge continues after corpus convergence")
        selected = min(counts, key=lambda pair: (-counts[pair], pair))
        frequency = counts[selected]
        if frequency < min_frequency:
            raise TokenizerError("stored merge falls below the frequency policy")
        if stored_pair != selected:
            raise TokenizerError("stored merge is not the canonical corpus pair")
        tokens_before = _weighted_token_count(streams)
        merged = _merge_weighted_streams(streams, selected, 256 + rank)
        tokens_after = _weighted_token_count(merged)
        applied_count = tokens_before - tokens_after
        if not 0 < applied_count <= frequency:
            raise TokenizerError("merge application count contradicts pair frequency")
        observations.append(
            MergeObservation(
                rank,
                selected,
                frequency,
                applied_count,
                tokens_before,
                tokens_after,
            )
        )
        streams = merged
    return tuple(observations)
