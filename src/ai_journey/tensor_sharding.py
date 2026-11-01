"""Reversible tensor parameter shards for a Hybrid Parallel Topology."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hybrid_topology import HybridParallelTopology


MAX_TENSOR_ELEMENTS = 1_000_000
MAX_TENSOR_DIMENSIONS = 8
MAX_TENSOR_SHARDS = 4_096


@dataclass(frozen=True)
class LogicalTensor:
    """One row-major logical tensor and its original dimensions."""

    shape: tuple[int, ...]
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise TypeError("logical tensor shape must be a tuple")
        if not self.shape:
            raise ValueError("logical tensor shape must be nonempty")
        if len(self.shape) > MAX_TENSOR_DIMENSIONS:
            raise ValueError("logical tensor shape exceeds the bounded dimensions")
        if any(type(dimension) is not int for dimension in self.shape):
            raise TypeError("logical tensor dimensions must be integers")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("logical tensor dimensions must be positive")
        if type(self.values) is not tuple:
            raise TypeError("logical tensor values must be a tuple")
        if len(self.values) > MAX_TENSOR_ELEMENTS:
            raise ValueError("logical tensor exceeds the bounded element count")
        if math.prod(self.shape) != len(self.values):
            raise ValueError("logical tensor shape must match its element count")


@dataclass(frozen=True)
class TensorShard:
    """The contiguous logical elements owned by one tensor rank."""

    tensor_rank: int
    tensor_parallel_size: int
    original_shape: tuple[int, ...]
    offset: int
    padded_length: int
    padding_elements: int
    values: tuple[object, ...]


def plan_tensor_shards(
    tensor: LogicalTensor, topology: HybridParallelTopology
) -> tuple[TensorShard, ...]:
    """Partition one logical tensor across the topology's tensor ranks."""

    if not isinstance(tensor, LogicalTensor):
        raise TypeError("tensor must be a LogicalTensor")
    if not isinstance(topology, HybridParallelTopology):
        raise TypeError("topology must be a HybridParallelTopology")
    tensor_size = topology.tensor_parallel_size
    if tensor_size > MAX_TENSOR_SHARDS:
        raise ValueError("topology exceeds the bounded tensor rank count")
    total = len(tensor.values)
    padded_length = math.ceil(total / tensor_size)
    quotient, remainder = divmod(total, tensor_size)
    shards = []
    offset = 0
    for rank in range(tensor_size):
        length = quotient + (rank < remainder)
        values = tensor.values[offset : offset + length]
        shards.append(
            TensorShard(
                tensor_rank=rank,
                tensor_parallel_size=tensor_size,
                original_shape=tensor.shape,
                offset=offset,
                padded_length=padded_length,
                padding_elements=padded_length - length,
                values=values,
            )
        )
        offset += length
    return tuple(shards)


def reconstruct_tensor(shards: tuple[TensorShard, ...]) -> LogicalTensor:
    """Reconstruct the original logical tensor from its canonical shards."""

    if type(shards) is not tuple:
        raise TypeError("shards must be a tuple")
    if not shards:
        raise ValueError("shards must be a nonempty canonical compatible collection")
    if len(shards) > MAX_TENSOR_SHARDS:
        raise ValueError("shards exceed the bounded tensor rank count")
    first = shards[0]
    if not isinstance(first, TensorShard):
        raise TypeError("shards must contain TensorShard values")
    if (
        type(first.original_shape) is not tuple
        or not first.original_shape
        or len(first.original_shape) > MAX_TENSOR_DIMENSIONS
        or any(type(dimension) is not int for dimension in first.original_shape)
        or any(dimension <= 0 for dimension in first.original_shape)
        or type(first.tensor_parallel_size) is not int
        or not 1 <= first.tensor_parallel_size <= MAX_TENSOR_SHARDS
        or len(shards) != first.tensor_parallel_size
    ):
        raise ValueError("shards must be a canonical compatible collection")
    total = math.prod(first.original_shape)
    if total > MAX_TENSOR_ELEMENTS:
        raise ValueError("shards exceed the bounded tensor element count")
    tensor_size = first.tensor_parallel_size
    padded_length = math.ceil(total / tensor_size)
    quotient, remainder = divmod(total, tensor_size)
    expected_offset = 0
    for rank, shard in enumerate(shards):
        length = quotient + (rank < remainder)
        if (
            not isinstance(shard, TensorShard)
            or type(shard.tensor_rank) is not int
            or type(shard.tensor_parallel_size) is not int
            or type(shard.offset) is not int
            or type(shard.padded_length) is not int
            or type(shard.padding_elements) is not int
            or type(shard.values) is not tuple
            or shard.tensor_rank != rank
            or shard.tensor_parallel_size != tensor_size
            or shard.original_shape != first.original_shape
            or shard.offset != expected_offset
            or shard.padded_length != padded_length
            or shard.padding_elements != padded_length - length
            or len(shard.values) != length
        ):
            raise ValueError("shards must be a canonical compatible collection")
        expected_offset += length
    return LogicalTensor(
        shape=first.original_shape,
        values=tuple(value for shard in shards for value in shard.values),
    )
