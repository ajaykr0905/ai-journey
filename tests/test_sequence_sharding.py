"""Sequence shard plans preserve ownership, order, and padding boundaries."""

from dataclasses import replace

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.sequence_sharding import (
    plan_sequence_shards,
    reconstruct_sequence_shards,
)


def test_even_sequence_shards_assign_every_position_once():
    topology = HybridParallelTopology(2, 1, 1, 1, 2)

    shards = plan_sequence_shards((11, 12, 13, 14), topology)

    assert tuple(
        (shard.sequence_rank, shard.start, shard.token_ids) for shard in shards
    ) == (
        (0, 0, (11, 12)),
        (1, 2, (13, 14)),
    )
    assert tuple(shard.padding_count for shard in shards) == (0, 0)
    assert all(shard.original_length == 4 for shard in shards)
    assert all(shard.padded_length == 4 for shard in shards)


def test_uneven_sequence_records_padding_without_reconstructing_it_as_input():
    topology = HybridParallelTopology(3, 1, 1, 1, 3)

    shards = plan_sequence_shards((21, 22, 23, 24, 25), topology)

    assert tuple((shard.token_ids, shard.padding_count) for shard in shards) == (
        ((21, 22), 0),
        ((23, 24), 0),
        ((25,), 1),
    )
    assert all(shard.original_length == 5 for shard in shards)
    assert all(shard.padded_length == 6 for shard in shards)
    assert reconstruct_sequence_shards(shards, topology) == (21, 22, 23, 24, 25)


def test_uneven_ownership_uses_deterministic_quotient_remainder_partitioning():
    topology = HybridParallelTopology(4, 1, 1, 1, 4)
    tokens = tuple(range(10))

    shards = plan_sequence_shards(tokens, topology)

    assert tuple((shard.start, len(shard.token_ids)) for shard in shards) == (
        (0, 3),
        (3, 3),
        (6, 2),
        (8, 2),
    )
    assert tuple(shard.padding_count for shard in shards) == (0, 0, 1, 1)
    assert reconstruct_sequence_shards(shards, topology) == tokens


def test_empty_and_more_ranks_than_tokens_preserve_exact_boundaries():
    topology = HybridParallelTopology(4, 1, 1, 1, 4)

    sparse = plan_sequence_shards((31, 32), topology)
    empty = plan_sequence_shards((), topology)

    assert tuple((shard.token_ids, shard.padding_count) for shard in sparse) == (
        ((31,), 0),
        ((32,), 0),
        ((), 1),
        ((), 1),
    )
    assert reconstruct_sequence_shards(sparse, topology) == (31, 32)
    assert tuple((shard.start, shard.padding_count) for shard in empty) == (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    )
    assert reconstruct_sequence_shards(empty, topology) == ()


@pytest.mark.parametrize("mutation", ["reordered", "duplicated", "missing"])
def test_reconstruction_rejects_noncanonical_shard_sets(mutation):
    topology = HybridParallelTopology(3, 1, 1, 1, 3)
    shards = plan_sequence_shards((41, 42, 43, 44), topology)
    invalid = {
        "reordered": (shards[1], shards[0], shards[2]),
        "duplicated": (shards[0], shards[0], shards[2]),
        "missing": shards[:-1],
    }[mutation]

    with pytest.raises(ValueError, match="canonical sequence-rank order"):
        reconstruct_sequence_shards(invalid, topology)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence_size", 4),
        ("original_length", 5),
        ("padded_length", 9),
        ("start", 1),
        ("padding_count", 1),
    ],
)
def test_reconstruction_rejects_shards_with_incompatible_boundaries(field, value):
    topology = HybridParallelTopology(3, 1, 1, 1, 3)
    shards = plan_sequence_shards((51, 52, 53, 54), topology)
    invalid = (replace(shards[0], **{field: value}), *shards[1:])

    with pytest.raises(ValueError, match="canonical sequence partition"):
        reconstruct_sequence_shards(invalid, topology)
