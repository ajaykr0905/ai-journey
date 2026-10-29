"""Analytical fully-sharded data-parallel planning primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .resource_accounting import DecoderSpec, bytes_per_element


class FSDPPlanningError(ValueError):
    """Raised when a sharding assumption is invalid."""


@dataclass(frozen=True)
class ShardSlice:
    """A contiguous byte interval from one named parameter group."""

    parameter_group: str
    offset_bytes: int
    length_bytes: int


@dataclass(frozen=True)
class ParameterShard:
    """The byte-exact flat-parameter partition assigned to one rank."""

    rank: int
    payload_bytes: int
    slices: tuple[ShardSlice, ...]


@dataclass(frozen=True)
class RankTrainingState:
    """Persistent sharded training-state bytes held by one rank."""

    rank: int
    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    master_parameters_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return the complete persistent state footprint for this rank."""

        return (
            self.parameters_bytes
            + self.gradients_bytes
            + self.optimizer_bytes
            + self.master_parameters_bytes
        )


@dataclass(frozen=True)
class RankAllGatherPeak:
    """Largest temporary full-parameter materialization on one rank."""

    rank: int
    parameter_group: str
    local_bytes: int
    materialized_bytes: int
    transient_bytes: int


def partition_parameter_groups(
    parameter_group_bytes: Mapping[str, int], *, world_size: int
) -> tuple[ParameterShard, ...]:
    """Partition an ordered flat parameter stream without padding or byte loss."""

    if type(world_size) is not int:
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise FSDPPlanningError("world_size must be positive")
    if not isinstance(parameter_group_bytes, Mapping) or not parameter_group_bytes:
        raise FSDPPlanningError("parameter_group_bytes must be a nonempty mapping")
    groups = []
    for name, size in parameter_group_bytes.items():
        if not isinstance(name, str) or not name:
            raise FSDPPlanningError("parameter group names must be nonempty strings")
        if type(size) is not int:
            raise TypeError("parameter group sizes must be integers")
        if size < 0:
            raise FSDPPlanningError("parameter group sizes must be nonnegative")
        if size:
            groups.append((name, size))
    total = sum(size for _, size in groups)
    if total == 0:
        raise FSDPPlanningError("at least one parameter group must contain bytes")

    targets = [
        total // world_size + (rank < total % world_size) for rank in range(world_size)
    ]
    shards = []
    group_index = 0
    group_offset = 0
    for rank, target in enumerate(targets):
        remaining = target
        slices = []
        while remaining:
            name, size = groups[group_index]
            length = min(remaining, size - group_offset)
            slices.append(ShardSlice(name, group_offset, length))
            remaining -= length
            group_offset += length
            if group_offset == size:
                group_index += 1
                group_offset = 0
        shards.append(ParameterShard(rank, target, tuple(slices)))
    return tuple(shards)


def decoder_parameter_shards(
    model: DecoderSpec, *, dtype: str, world_size: int
) -> tuple[ParameterShard, ...]:
    """Partition a decoder's declared parameter inventory across ranks."""

    if not isinstance(model, DecoderSpec):
        raise TypeError("model must be a DecoderSpec")
    width = bytes_per_element(dtype)
    groups = {name: count * width for name, count in model.parameter_groups().items()}
    return partition_parameter_groups(groups, world_size=world_size)


def all_gather_transient_peaks(
    parameter_group_bytes: Mapping[str, int],
    shards: tuple[ParameterShard, ...],
) -> tuple[RankAllGatherPeak, ...]:
    """Find each rank's largest missing group payload during FSDP all-gather."""

    if not shards:
        raise FSDPPlanningError("shards must be nonempty")
    expected = partition_parameter_groups(parameter_group_bytes, world_size=len(shards))
    if shards != expected:
        raise FSDPPlanningError(
            "shards must match the canonical partition of parameter_group_bytes"
        )
    nonempty_groups = {
        name: size for name, size in parameter_group_bytes.items() if size
    }
    peaks = []
    for shard in shards:
        local_by_group = {name: 0 for name in nonempty_groups}
        for part in shard.slices:
            local_by_group[part.parameter_group] += part.length_bytes
        group, size = max(
            nonempty_groups.items(),
            key=lambda item: (item[1] - local_by_group[item[0]], item[1]),
        )
        local = local_by_group[group]
        peaks.append(
            RankAllGatherPeak(
                rank=shard.rank,
                parameter_group=group,
                local_bytes=local,
                materialized_bytes=size,
                transient_bytes=size - local,
            )
        )
    return tuple(peaks)


def sharded_training_state(
    model: DecoderSpec,
    *,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_dtype: str,
    optimizer_slots: int,
    world_size: int,
    master_parameter_dtype: str | None = None,
) -> tuple[RankTrainingState, ...]:
    """Inventory persistent FSDP state without assuming measured allocation."""

    if type(optimizer_slots) is not int:
        raise TypeError("optimizer_slots must be an integer")
    if optimizer_slots < 0:
        raise FSDPPlanningError("optimizer_slots must be nonnegative")
    parameter_width = bytes_per_element(parameter_dtype)
    gradient_width = bytes_per_element(gradient_dtype)
    optimizer_width = bytes_per_element(optimizer_dtype)
    master_width = (
        0
        if master_parameter_dtype is None
        else bytes_per_element(master_parameter_dtype)
    )
    count_shards = partition_parameter_groups(
        model.parameter_groups(), world_size=world_size
    )
    return tuple(
        RankTrainingState(
            rank=shard.rank,
            parameters_bytes=shard.payload_bytes * parameter_width,
            gradients_bytes=shard.payload_bytes * gradient_width,
            optimizer_bytes=shard.payload_bytes * optimizer_width * optimizer_slots,
            master_parameters_bytes=shard.payload_bytes * master_width,
        )
        for shard in count_shards
    )
