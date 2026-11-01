"""Deterministic analytical plans for sequence-parallel token ownership."""

from __future__ import annotations

from dataclasses import dataclass

from .hybrid_topology import HybridParallelTopology


@dataclass(frozen=True)
class SequenceShard:
    """One sequence rank's immutable token ownership and padding boundary."""

    sequence_rank: int
    sequence_size: int
    original_length: int
    padded_length: int
    start: int
    token_ids: tuple[int, ...]
    padding_count: int


def plan_sequence_shards(
    token_ids: tuple[int, ...], topology: HybridParallelTopology
) -> tuple[SequenceShard, ...]:
    """Assign token positions to equal-width sequence-rank shards."""

    sequence_size = topology.sequence_parallel_size
    original_length = len(token_ids)
    base_width, remainder = divmod(original_length, sequence_size)
    padded_width = base_width + bool(remainder)
    padded_length = padded_width * sequence_size
    shards = []
    for sequence_rank in range(sequence_size):
        start = sequence_rank * base_width + min(sequence_rank, remainder)
        token_count = base_width + (sequence_rank < remainder)
        owned = token_ids[start : start + token_count]
        shards.append(
            SequenceShard(
                sequence_rank=sequence_rank,
                sequence_size=sequence_size,
                original_length=original_length,
                padded_length=padded_length,
                start=start,
                token_ids=owned,
                padding_count=padded_width - len(owned),
            )
        )
    return tuple(shards)


def reconstruct_sequence_shards(
    shards: tuple[SequenceShard, ...], topology: HybridParallelTopology
) -> tuple[int, ...]:
    """Reconstruct original token order while excluding declared padding."""

    expected_ranks = tuple(range(topology.sequence_parallel_size))
    actual_ranks = tuple(shard.sequence_rank for shard in shards)
    if actual_ranks != expected_ranks:
        raise ValueError("shards must use canonical sequence-rank order")
    original_length = shards[0].original_length
    sequence_size = topology.sequence_parallel_size
    if type(original_length) is not int or original_length < 0:
        raise ValueError("shards must match the canonical sequence partition")
    base_width, remainder = divmod(original_length, sequence_size)
    padded_width = base_width + bool(remainder)
    padded_length = padded_width * sequence_size
    for sequence_rank, shard in enumerate(shards):
        start = sequence_rank * base_width + min(sequence_rank, remainder)
        token_count = base_width + (sequence_rank < remainder)
        expected = (
            sequence_size,
            original_length,
            padded_length,
            start,
            token_count,
            padded_width - token_count,
        )
        actual = (
            shard.sequence_size,
            shard.original_length,
            shard.padded_length,
            shard.start,
            len(shard.token_ids),
            shard.padding_count,
        )
        if actual != expected:
            raise ValueError("shards must match the canonical sequence partition")
    return tuple(token for shard in shards for token in shard.token_ids)
