"""Tensor shard plans preserve each bounded logical parameter element."""

from dataclasses import replace

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.tensor_sharding import (
    LogicalTensor,
    plan_tensor_shards,
    reconstruct_tensor,
)


def test_even_tensor_shards_follow_the_topology_tensor_dimension():
    topology = HybridParallelTopology(4, 1, 2, 2, 1)
    tensor = LogicalTensor(shape=(2, 4), values=tuple(range(8)))

    shards = plan_tensor_shards(tensor, topology)

    assert [shard.tensor_rank for shard in shards] == [0, 1]
    assert [shard.values for shard in shards] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert all(shard.original_shape == (2, 4) for shard in shards)


def test_uneven_shards_record_padding_and_reconstruct_the_original_tensor():
    topology = HybridParallelTopology(3, 1, 3, 1, 1)
    tensor = LogicalTensor(shape=(2, 5), values=tuple(range(10)))

    shards = plan_tensor_shards(tensor, topology)

    assert [shard.offset for shard in shards] == [0, 4, 7]
    assert [len(shard.values) for shard in shards] == [4, 3, 3]
    assert [shard.padded_length for shard in shards] == [4, 4, 4]
    assert [shard.padding_elements for shard in shards] == [0, 1, 1]
    assert reconstruct_tensor(shards) == tensor


@pytest.mark.parametrize(
    ("shape", "values", "message"),
    [
        ((2, 2), (0, 1, 2), "shape must match"),
        ((2, 0), (), "dimensions must be positive"),
        ((2, True), (0, 1), "dimensions must be integers"),
        ((), (), "shape must be nonempty"),
    ],
)
def test_logical_tensor_rejects_incoherent_dimensions(shape, values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        LogicalTensor(shape=shape, values=values)


def test_logical_tensor_rejects_unbounded_element_work(monkeypatch):
    monkeypatch.setattr("ai_journey.tensor_sharding.MAX_TENSOR_ELEMENTS", 2)

    with pytest.raises(ValueError, match="bounded element count"):
        LogicalTensor(shape=(3,), values=(0, 1, 2))


def test_more_tensor_ranks_than_elements_preserve_empty_rank_ownership():
    topology = HybridParallelTopology(5, 1, 5, 1, 1)
    tensor = LogicalTensor(shape=(2,), values=("weight-0", "weight-1"))

    shards = plan_tensor_shards(tensor, topology)

    assert [shard.values for shard in shards] == [
        ("weight-0",),
        ("weight-1",),
        (),
        (),
        (),
    ]
    assert [shard.offset for shard in shards] == [0, 1, 2, 2, 2]
    assert [shard.padding_elements for shard in shards] == [0, 0, 1, 1, 1]
    assert reconstruct_tensor(shards) == tensor


@pytest.mark.parametrize(
    "mutation",
    ["reordered", "duplicate-rank", "shape", "offset", "parallel-size", "payload"],
)
def test_reconstruction_rejects_incompatible_shard_collections(mutation):
    topology = HybridParallelTopology(3, 1, 3, 1, 1)
    shards = plan_tensor_shards(
        LogicalTensor(shape=(5,), values=(0, 1, 2, 3, 4)), topology
    )
    if mutation == "reordered":
        incompatible = tuple(reversed(shards))
    elif mutation == "duplicate-rank":
        incompatible = (shards[0], replace(shards[1], tensor_rank=0), shards[2])
    elif mutation == "shape":
        incompatible = (shards[0], replace(shards[1], original_shape=(1, 5)), shards[2])
    elif mutation == "offset":
        incompatible = (
            shards[0],
            replace(shards[1], offset=shards[1].offset + 1),
            shards[2],
        )
    elif mutation == "parallel-size":
        incompatible = (replace(shards[0], tensor_parallel_size=0), *shards[1:])
    else:
        incompatible = (shards[0], replace(shards[1], values=[2, 3]), shards[2])

    with pytest.raises(ValueError, match="canonical compatible collection"):
        reconstruct_tensor(incompatible)


def test_reconstruction_rejects_an_empty_collection():
    with pytest.raises(ValueError, match="nonempty"):
        reconstruct_tensor(())


def test_planning_rejects_unbounded_tensor_rank_work(monkeypatch):
    monkeypatch.setattr("ai_journey.tensor_sharding.MAX_TENSOR_SHARDS", 2)
    topology = HybridParallelTopology(3, 1, 3, 1, 1)

    with pytest.raises(ValueError, match="bounded tensor rank count"):
        plan_tensor_shards(LogicalTensor((1,), (0,)), topology)
