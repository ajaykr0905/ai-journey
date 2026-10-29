import pytest

from ai_journey.distributed_capacity import (
    ClusterTopology,
    DistributedCapacityError,
    LinkProfile,
    ring_all_reduce,
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


def test_ring_all_reduce_accounts_for_padding_steps_and_link_scope():
    single_node = ClusterTopology(4, 4, link(100.0, 0.01), link(10.0, 0.1))
    estimate = ring_all_reduce(10, single_node)

    assert estimate.operation == "all_reduce"
    assert estimate.steps == 6
    assert estimate.wire_bytes_per_rank == 18
    assert estimate.seconds == pytest.approx(0.24)
    assert estimate.link_scope == "intra_node"

    multi_node = ClusterTopology(4, 2, link(100.0, 0.01), link(10.0, 0.1))
    assert ring_all_reduce(10, multi_node).seconds == pytest.approx(2.4)
    assert ring_all_reduce(10, multi_node).link_scope == "inter_node"


def test_ring_all_reduce_treats_one_rank_as_zero_wire_work():
    topology = ClusterTopology(1, 1, link(), link())
    estimate = ring_all_reduce(128, topology)
    assert estimate.steps == estimate.wire_bytes_per_rank == 0
    assert estimate.seconds == 0
    with pytest.raises(TypeError, match="payload_bytes"):
        ring_all_reduce(True, topology)
