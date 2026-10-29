import pytest

from ai_journey.distributed_capacity import (
    ClusterTopology,
    DistributedCapacityError,
    GradientBucket,
    LinkProfile,
    bucketize_gradients,
    gradient_sync_schedule,
    hierarchical_all_reduce,
    ring_all_gather,
    ring_all_reduce,
    ring_reduce_scatter,
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


def test_ring_reduce_scatter_and_all_gather_expose_one_phase_costs():
    topology = ClusterTopology(4, 4, link(100.0, 0.01), link())

    scatter = ring_reduce_scatter(10, topology)
    gather = ring_all_gather(3, topology)

    assert scatter.operation == "reduce_scatter"
    assert gather.operation == "all_gather"
    assert scatter.steps == gather.steps == 3
    assert scatter.wire_bytes_per_rank == gather.wire_bytes_per_rank == 9
    assert scatter.seconds == gather.seconds == pytest.approx(0.12)


def test_one_rank_collective_phases_have_no_wire_cost():
    topology = ClusterTopology(1, 1, link(), link())
    assert ring_reduce_scatter(8, topology).seconds == 0
    assert ring_all_gather(8, topology).wire_bytes_per_rank == 0


def test_hierarchical_all_reduce_uses_local_shards_across_nodes():
    topology = ClusterTopology(8, 4, link(100.0, 0.01), link(10.0, 0.1))
    estimate = hierarchical_all_reduce(32, topology)

    assert estimate.local_reduce_scatter.wire_bytes_per_rank == 24
    assert estimate.cross_node_all_reduce.payload_bytes == 8
    assert estimate.cross_node_all_reduce.wire_bytes_per_rank == 8
    assert estimate.local_all_gather.wire_bytes_per_rank == 24
    assert estimate.wire_bytes_per_rank == 56
    assert estimate.seconds == pytest.approx(1.54)
    assert estimate.seconds < ring_all_reduce(32, topology).seconds


def test_hierarchical_all_reduce_degenerates_to_local_ring_on_one_node():
    topology = ClusterTopology(4, 4, link(100.0, 0.01), link(10.0, 0.1))
    hierarchical = hierarchical_all_reduce(32, topology)
    flat = ring_all_reduce(32, topology)
    assert hierarchical.seconds == flat.seconds
    assert hierarchical.wire_bytes_per_rank == flat.wire_bytes_per_rank


def test_gradient_buckets_preserve_order_offsets_and_every_byte():
    buckets = bucketize_gradients(
        {"embedding": 3, "blocks": 10, "unused": 0, "head": 2}, bucket_bytes=6
    )

    assert [bucket.payload_bytes for bucket in buckets] == [6, 6, 3]
    assert [bucket.index for bucket in buckets] == [0, 1, 2]
    assert [
        (part.parameter_group, part.offset_bytes, part.length_bytes)
        for bucket in buckets
        for part in bucket.slices
    ] == [
        ("embedding", 0, 3),
        ("blocks", 0, 3),
        ("blocks", 3, 6),
        ("blocks", 9, 1),
        ("head", 0, 2),
    ]
    assert sum(bucket.payload_bytes for bucket in buckets) == 15


def test_gradient_bucket_admission_rejects_empty_and_ambiguous_groups():
    with pytest.raises(DistributedCapacityError, match="nonempty mapping"):
        bucketize_gradients({}, bucket_bytes=1)
    with pytest.raises(DistributedCapacityError, match="contain bytes"):
        bucketize_gradients({"zero": 0}, bucket_bytes=1)
    with pytest.raises(TypeError, match="sizes"):
        bucketize_gradients({"bad": True}, bucket_bytes=1)


def test_gradient_sync_schedule_totals_each_bucket_collective():
    topology = ClusterTopology(8, 4, link(100.0, 0.01), link(10.0, 0.1))
    buckets = bucketize_gradients({"model": 64, "head": 8}, bucket_bytes=32)

    flat = gradient_sync_schedule(buckets, topology, algorithm="flat_ring")
    hierarchical = gradient_sync_schedule(
        buckets, topology, algorithm="hierarchical_ring"
    )

    assert [item.payload_bytes for item in flat.buckets] == [32, 32, 8]
    assert flat.wire_bytes_per_rank == sum(
        ring_all_reduce(size, topology).wire_bytes_per_rank for size in (32, 32, 8)
    )
    assert hierarchical.seconds < flat.seconds
    assert hierarchical.algorithm == "hierarchical_ring"


def test_gradient_sync_schedule_rejects_tampered_or_reordered_buckets():
    topology = ClusterTopology(2, 2, link(), link())
    buckets = bucketize_gradients({"model": 4}, bucket_bytes=4)
    tampered = GradientBucket(0, 3, buckets[0].slices)
    with pytest.raises(DistributedCapacityError, match="payload"):
        gradient_sync_schedule((tampered,), topology, algorithm="flat_ring")
    with pytest.raises(DistributedCapacityError, match="unsupported"):
        gradient_sync_schedule(buckets, topology, algorithm="tree")
