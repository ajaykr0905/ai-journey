"""Analytical distributed-training capacity primitives.

These helpers model declared topology and link assumptions. They do not probe
hardware, execute collectives, or claim measured network performance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


class DistributedCapacityError(ValueError):
    """Raised when a distributed capacity assumption is invalid."""


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise DistributedCapacityError(f"{name} must be positive")
    return value


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DistributedCapacityError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class LinkProfile:
    """One directional link model used for analytical collectives."""

    bandwidth_bytes_per_second: float
    latency_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bandwidth_bytes_per_second",
            _positive_finite(
                self.bandwidth_bytes_per_second, "bandwidth_bytes_per_second"
            ),
        )
        object.__setattr__(
            self,
            "latency_seconds",
            _positive_finite(self.latency_seconds, "latency_seconds"),
        )


@dataclass(frozen=True)
class ClusterTopology:
    """A homogeneous worker topology with explicit local and remote links."""

    world_size: int
    workers_per_node: int
    intra_node: LinkProfile
    inter_node: LinkProfile

    def __post_init__(self) -> None:
        world = _positive_integer(self.world_size, "world_size")
        local = _positive_integer(self.workers_per_node, "workers_per_node")
        if world % local:
            raise DistributedCapacityError(
                "world_size must be divisible by workers_per_node"
            )
        if not isinstance(self.intra_node, LinkProfile):
            raise TypeError("intra_node must be a LinkProfile")
        if not isinstance(self.inter_node, LinkProfile):
            raise TypeError("inter_node must be a LinkProfile")

    @property
    def node_count(self) -> int:
        return self.world_size // self.workers_per_node

    @property
    def crosses_nodes(self) -> bool:
        return self.node_count > 1

    @property
    def collective_link(self) -> LinkProfile:
        """Return the conservative link used by a flat cross-rank ring."""

        return self.inter_node if self.crosses_nodes else self.intra_node


@dataclass(frozen=True)
class CollectiveEstimate:
    """Per-rank wire work and lower-bound time for one collective."""

    operation: str
    payload_bytes: int
    world_size: int
    steps: int
    wire_bytes_per_rank: int
    seconds: float
    link_scope: str


@dataclass(frozen=True)
class HierarchicalEstimate:
    """Per-rank phases for a two-level all-reduce."""

    payload_bytes: int
    local_reduce_scatter: CollectiveEstimate
    cross_node_all_reduce: CollectiveEstimate
    local_all_gather: CollectiveEstimate
    seconds: float
    wire_bytes_per_rank: int


@dataclass(frozen=True)
class BucketSlice:
    """A contiguous byte range from one named gradient group."""

    parameter_group: str
    offset_bytes: int
    length_bytes: int


@dataclass(frozen=True)
class GradientBucket:
    """One ordered collective payload assembled from gradient slices."""

    index: int
    payload_bytes: int
    slices: tuple[BucketSlice, ...]


@dataclass(frozen=True)
class BucketSync:
    """Collective cost assigned to one gradient bucket."""

    bucket_index: int
    payload_bytes: int
    wire_bytes_per_rank: int
    seconds: float


@dataclass(frozen=True)
class GradientSyncSchedule:
    """Ordered communication schedule for one backward pass."""

    algorithm: str
    buckets: tuple[BucketSync, ...]
    wire_bytes_per_rank: int
    seconds: float


@dataclass(frozen=True)
class OverlapEstimate:
    """Serialized communication overlap against a backward timeline."""

    backward_seconds: float
    communication_seconds: float
    overlapped_seconds: float
    exposed_seconds: float
    step_seconds: float
    bucket_finish_seconds: tuple[float, ...]


def ring_all_reduce(
    payload_bytes: int, topology: ClusterTopology
) -> CollectiveEstimate:
    """Estimate a flat ring reduce-scatter plus all-gather.

    Payloads are split into equal ceiling-sized chunks so non-divisible tensors
    cannot silently undercount padding on the wire.
    """

    payload = _positive_integer(payload_bytes, "payload_bytes")
    if not isinstance(topology, ClusterTopology):
        raise TypeError("topology must be a ClusterTopology")
    if topology.world_size == 1:
        return CollectiveEstimate("all_reduce", payload, 1, 0, 0, 0.0, "local")
    chunk = math.ceil(payload / topology.world_size)
    steps = 2 * (topology.world_size - 1)
    wire_bytes = steps * chunk
    link = topology.collective_link
    seconds = (
        wire_bytes / link.bandwidth_bytes_per_second + steps * link.latency_seconds
    )
    return CollectiveEstimate(
        "all_reduce",
        payload,
        topology.world_size,
        steps,
        wire_bytes,
        seconds,
        "inter_node" if topology.crosses_nodes else "intra_node",
    )


def _ring_phase_estimate(
    operation: str,
    payload_bytes: int,
    topology: ClusterTopology,
    *,
    payload_is_shard: bool,
) -> CollectiveEstimate:
    payload = _positive_integer(payload_bytes, "payload_bytes")
    if not isinstance(topology, ClusterTopology):
        raise TypeError("topology must be a ClusterTopology")
    if topology.world_size == 1:
        return CollectiveEstimate(operation, payload, 1, 0, 0, 0.0, "local")
    chunk = payload if payload_is_shard else math.ceil(payload / topology.world_size)
    steps = topology.world_size - 1
    wire_bytes = steps * chunk
    link = topology.collective_link
    seconds = (
        wire_bytes / link.bandwidth_bytes_per_second + steps * link.latency_seconds
    )
    return CollectiveEstimate(
        operation,
        payload,
        topology.world_size,
        steps,
        wire_bytes,
        seconds,
        "inter_node" if topology.crosses_nodes else "intra_node",
    )


def ring_reduce_scatter(
    payload_bytes: int, topology: ClusterTopology
) -> CollectiveEstimate:
    """Estimate one ring reduce-scatter over a full unsharded payload."""

    return _ring_phase_estimate(
        "reduce_scatter", payload_bytes, topology, payload_is_shard=False
    )


def ring_all_gather(shard_bytes: int, topology: ClusterTopology) -> CollectiveEstimate:
    """Estimate one ring all-gather from an equal per-rank shard."""

    return _ring_phase_estimate(
        "all_gather", shard_bytes, topology, payload_is_shard=True
    )


def hierarchical_all_reduce(
    payload_bytes: int, topology: ClusterTopology
) -> HierarchicalEstimate:
    """Estimate local scatter, cross-node shard reduction, and local gather."""

    payload = _positive_integer(payload_bytes, "payload_bytes")
    if not isinstance(topology, ClusterTopology):
        raise TypeError("topology must be a ClusterTopology")
    local_topology = ClusterTopology(
        topology.workers_per_node,
        topology.workers_per_node,
        topology.intra_node,
        topology.intra_node,
    )
    local_scatter = ring_reduce_scatter(payload, local_topology)
    shard_bytes = math.ceil(payload / topology.workers_per_node)
    node_topology = ClusterTopology(
        topology.node_count,
        1,
        topology.inter_node,
        topology.inter_node,
    )
    cross_node = ring_all_reduce(shard_bytes, node_topology)
    local_gather = ring_all_gather(shard_bytes, local_topology)
    phases = (local_scatter, cross_node, local_gather)
    return HierarchicalEstimate(
        payload,
        local_scatter,
        cross_node,
        local_gather,
        math.fsum(phase.seconds for phase in phases),
        sum(phase.wire_bytes_per_rank for phase in phases),
    )


def bucketize_gradients(
    parameter_group_bytes: Mapping[str, int], *, bucket_bytes: int
) -> tuple[GradientBucket, ...]:
    """Split ordered gradient groups into bounded collective payloads."""

    limit = _positive_integer(bucket_bytes, "bucket_bytes")
    if not isinstance(parameter_group_bytes, Mapping) or not parameter_group_bytes:
        raise DistributedCapacityError(
            "parameter_group_bytes must be a nonempty mapping"
        )
    normalized = []
    for name, size in parameter_group_bytes.items():
        if not isinstance(name, str) or not name:
            raise DistributedCapacityError(
                "gradient group names must be nonempty strings"
            )
        if type(size) is not int:
            raise TypeError("gradient group sizes must be integers")
        if size < 0:
            raise DistributedCapacityError("gradient group sizes must be nonnegative")
        if size:
            normalized.append((name, size))
    if not normalized:
        raise DistributedCapacityError("at least one gradient group must contain bytes")

    buckets: list[GradientBucket] = []
    slices: list[BucketSlice] = []
    filled = 0
    for name, size in normalized:
        offset = 0
        while offset < size:
            length = min(limit - filled, size - offset)
            slices.append(BucketSlice(name, offset, length))
            filled += length
            offset += length
            if filled == limit:
                buckets.append(GradientBucket(len(buckets), filled, tuple(slices)))
                slices, filled = [], 0
    if slices:
        buckets.append(GradientBucket(len(buckets), filled, tuple(slices)))
    return tuple(buckets)


def gradient_sync_schedule(
    buckets: tuple[GradientBucket, ...],
    topology: ClusterTopology,
    *,
    algorithm: str,
) -> GradientSyncSchedule:
    """Estimate ordered bucket collectives for flat or hierarchical rings."""

    if not isinstance(buckets, tuple) or not buckets:
        raise DistributedCapacityError("buckets must be a nonempty tuple")
    if not isinstance(topology, ClusterTopology):
        raise TypeError("topology must be a ClusterTopology")
    if algorithm not in {"flat_ring", "hierarchical_ring"}:
        raise DistributedCapacityError("unsupported gradient synchronization algorithm")
    estimates = []
    for expected_index, bucket in enumerate(buckets):
        if not isinstance(bucket, GradientBucket) or bucket.index != expected_index:
            raise DistributedCapacityError(
                "gradient buckets must be ordered and contiguous"
            )
        if bucket.payload_bytes != sum(part.length_bytes for part in bucket.slices):
            raise DistributedCapacityError(
                "gradient bucket payload does not match its slices"
            )
        collective = (
            ring_all_reduce(bucket.payload_bytes, topology)
            if algorithm == "flat_ring"
            else hierarchical_all_reduce(bucket.payload_bytes, topology)
        )
        estimates.append(
            BucketSync(
                bucket.index,
                bucket.payload_bytes,
                collective.wire_bytes_per_rank,
                collective.seconds,
            )
        )
    return GradientSyncSchedule(
        algorithm,
        tuple(estimates),
        sum(estimate.wire_bytes_per_rank for estimate in estimates),
        math.fsum(estimate.seconds for estimate in estimates),
    )


def estimate_backward_overlap(
    schedule: GradientSyncSchedule,
    *,
    backward_seconds: float,
    release_fractions: tuple[float, ...],
) -> OverlapEstimate:
    """Place ready gradient buckets on one serialized communication stream."""

    if not isinstance(schedule, GradientSyncSchedule):
        raise TypeError("schedule must be a GradientSyncSchedule")
    backward = _positive_finite(backward_seconds, "backward_seconds")
    if not isinstance(release_fractions, tuple) or len(release_fractions) != len(
        schedule.buckets
    ):
        raise DistributedCapacityError(
            "release_fractions must match the gradient bucket count"
        )
    previous = -1.0
    finish = 0.0
    finishes = []
    for fraction, bucket in zip(release_fractions, schedule.buckets, strict=True):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
            or fraction < previous
        ):
            raise DistributedCapacityError(
                "release fractions must be finite, ordered, and in [0, 1]"
            )
        start = max(finish, backward * fraction)
        finish = start + bucket.seconds
        finishes.append(finish)
        previous = float(fraction)
    step = max(backward, finish)
    exposed = step - backward
    overlapped = max(0.0, schedule.seconds - exposed)
    return OverlapEstimate(
        backward,
        schedule.seconds,
        overlapped,
        exposed,
        step,
        tuple(finishes),
    )
