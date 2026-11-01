"""Hybrid parallel topology contracts expose one deterministic rank space."""

from dataclasses import FrozenInstanceError

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology, RankCoordinate


def test_topology_is_immutable_and_has_a_canonical_public_representation():
    topology = HybridParallelTopology(
        world_size=8,
        data_parallel_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        sequence_parallel_size=1,
    )

    assert topology.to_mapping() == {
        "world_size": 8,
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "sequence_parallel_size": 1,
    }
    assert HybridParallelTopology.from_mapping(topology.to_mapping()) == topology
    with pytest.raises(FrozenInstanceError):
        topology.world_size = 16


@pytest.mark.parametrize(
    "field",
    [
        "world_size",
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "sequence_parallel_size",
    ],
)
def test_topology_rejects_nonpositive_dimensions(field):
    values = {
        "world_size": 8,
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "sequence_parallel_size": 1,
    }
    values[field] = 0

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        HybridParallelTopology(**values)


def test_topology_rejects_noninteger_and_mismatched_world_sizes():
    with pytest.raises(TypeError, match="data_parallel_size must be an integer"):
        HybridParallelTopology(8, True, 2, 2, 1)
    with pytest.raises(ValueError, match="product must equal world_size"):
        HybridParallelTopology(16, 2, 2, 2, 1)


def test_composed_rank_coordinates_are_bijective_and_sequence_major():
    topology = HybridParallelTopology(16, 2, 2, 2, 2)
    expected = {
        0: RankCoordinate(data=0, tensor=0, pipeline=0, sequence=0),
        1: RankCoordinate(data=0, tensor=0, pipeline=0, sequence=1),
        2: RankCoordinate(data=0, tensor=0, pipeline=1, sequence=0),
        4: RankCoordinate(data=0, tensor=1, pipeline=0, sequence=0),
        8: RankCoordinate(data=1, tensor=0, pipeline=0, sequence=0),
        15: RankCoordinate(data=1, tensor=1, pipeline=1, sequence=1),
    }

    assert {rank: topology.coordinate_for_rank(rank) for rank in expected} == expected
    coordinates = tuple(
        topology.coordinate_for_rank(rank) for rank in range(topology.world_size)
    )
    assert len(set(coordinates)) == topology.world_size
    assert tuple(topology.rank_for_coordinate(item) for item in coordinates) == tuple(
        range(topology.world_size)
    )


def test_one_rank_topology_has_the_origin_coordinate():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)

    assert topology.coordinate_for_rank(0) == RankCoordinate(0, 0, 0, 0)
    assert topology.rank_for_coordinate(RankCoordinate(0, 0, 0, 0)) == 0


def test_composed_topology_derives_stable_peer_groups_for_every_dimension():
    topology = HybridParallelTopology(8, 2, 2, 2, 1)

    assert topology.peer_groups("data") == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    assert topology.peer_groups("tensor") == (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    )
    assert topology.peer_groups("pipeline") == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    )
    assert topology.peer_groups("sequence") == tuple((rank,) for rank in range(8))
    assert topology.peer_group(5, "data") == (1, 5)
    assert topology.peer_group(5, "tensor") == (5, 7)
    assert topology.peer_group(5, "pipeline") == (4, 5)
    assert topology.peer_group(5, "sequence") == (5,)


def test_one_dimensional_topology_assigns_every_rank_to_exactly_one_peer_group():
    topology = HybridParallelTopology(4, 4, 1, 1, 1)

    assert topology.peer_groups("data") == ((0, 1, 2, 3),)
    for dimension in ("data", "tensor", "pipeline", "sequence"):
        groups = topology.peer_groups(dimension)
        assert sorted(rank for group in groups for rank in group) == list(range(4))


def test_rank_and_group_queries_reject_values_outside_the_topology():
    topology = HybridParallelTopology(8, 2, 2, 2, 1)

    with pytest.raises(ValueError, match="rank is outside"):
        topology.coordinate_for_rank(8)
    with pytest.raises(TypeError, match="rank must be an integer"):
        topology.coordinate_for_rank(True)
    with pytest.raises(ValueError, match="coordinate is outside"):
        topology.rank_for_coordinate(RankCoordinate(2, 0, 0, 0))
    with pytest.raises(ValueError, match="dimension must be"):
        topology.peer_groups("model")


def test_canonical_topology_input_rejects_missing_or_unknown_fields():
    valid = HybridParallelTopology(8, 2, 2, 2, 1).to_mapping()

    with pytest.raises(ValueError, match="topology fields do not match"):
        HybridParallelTopology.from_mapping({**valid, "backend": "gloo"})
    missing = dict(valid)
    missing.pop("sequence_parallel_size")
    with pytest.raises(ValueError, match="topology fields do not match"):
        HybridParallelTopology.from_mapping(missing)
