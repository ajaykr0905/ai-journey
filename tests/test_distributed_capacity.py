import pytest

from ai_journey.distributed_capacity import (
    ClusterTopology,
    DistributedCapacityError,
    LinkProfile,
)


def link(bandwidth=100.0, latency=0.001):
    return LinkProfile(bandwidth, latency)


def test_cluster_topology_exposes_node_geometry_and_cross_node_boundary():
    single = ClusterTopology(8, 8, link(), link(10.0, 0.01))
    multi = ClusterTopology(16, 8, link(), link(10.0, 0.01))

    assert single.node_count == 1
    assert single.crosses_nodes is False
    assert multi.node_count == 2
    assert multi.crosses_nodes is True


def test_cluster_topology_rejects_partial_nodes_and_ambiguous_numeric_values():
    with pytest.raises(DistributedCapacityError, match="divisible"):
        ClusterTopology(10, 8, link(), link())
    with pytest.raises(TypeError, match="world_size"):
        ClusterTopology(True, 1, link(), link())
    with pytest.raises(DistributedCapacityError, match="positive and finite"):
        link(float("inf"))
    with pytest.raises(TypeError, match="intra_node"):
        ClusterTopology(1, 1, object(), link())
