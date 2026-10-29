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
