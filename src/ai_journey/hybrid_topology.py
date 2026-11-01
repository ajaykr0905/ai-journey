"""Deterministic contracts for a bounded Hybrid Parallel Topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


ParallelDimension = Literal["data", "tensor", "pipeline", "sequence"]


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RankCoordinate:
    """One global rank's coordinate in the four-dimensional rank space."""

    data: int
    tensor: int
    pipeline: int
    sequence: int

    def __post_init__(self) -> None:
        for name in ("data", "tensor", "pipeline", "sequence"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} coordinate must be an integer")
            if value < 0:
                raise ValueError(f"{name} coordinate must be nonnegative")


@dataclass(frozen=True)
class HybridParallelTopology:
    """Factor participating ranks across four parallel dimensions."""

    world_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    sequence_parallel_size: int

    def __post_init__(self) -> None:
        dimensions = {
            "world_size": self.world_size,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel_size": self.sequence_parallel_size,
        }
        for name, value in dimensions.items():
            _positive_integer(value, name)
        product = (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.sequence_parallel_size
        )
        if product != self.world_size:
            raise ValueError("parallel dimension product must equal world_size")

    def coordinate_for_rank(self, rank: int) -> RankCoordinate:
        """Decode one admitted global rank using sequence-major ordering."""

        if type(rank) is not int:
            raise TypeError("rank must be an integer")
        if not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the topology")
        remaining = rank
        sequence = remaining % self.sequence_parallel_size
        remaining //= self.sequence_parallel_size
        pipeline = remaining % self.pipeline_parallel_size
        remaining //= self.pipeline_parallel_size
        tensor = remaining % self.tensor_parallel_size
        data = remaining // self.tensor_parallel_size
        return RankCoordinate(data, tensor, pipeline, sequence)

    def rank_for_coordinate(self, coordinate: RankCoordinate) -> int:
        """Encode an admitted rank coordinate into its unique global rank."""

        if not isinstance(coordinate, RankCoordinate):
            raise TypeError("coordinate must be a RankCoordinate")
        bounds = (
            (coordinate.data, self.data_parallel_size),
            (coordinate.tensor, self.tensor_parallel_size),
            (coordinate.pipeline, self.pipeline_parallel_size),
            (coordinate.sequence, self.sequence_parallel_size),
        )
        if any(value >= size for value, size in bounds):
            raise ValueError("coordinate is outside the topology")
        return (
            (coordinate.data * self.tensor_parallel_size + coordinate.tensor)
            * self.pipeline_parallel_size
            + coordinate.pipeline
        ) * self.sequence_parallel_size + coordinate.sequence

    def peer_group(self, rank: int, dimension: ParallelDimension) -> tuple[int, ...]:
        """Return ranks sharing all coordinates except one dimension."""

        coordinate = self.coordinate_for_rank(rank)
        sizes = {
            "data": self.data_parallel_size,
            "tensor": self.tensor_parallel_size,
            "pipeline": self.pipeline_parallel_size,
            "sequence": self.sequence_parallel_size,
        }
        if dimension not in sizes:
            raise ValueError("dimension must be data, tensor, pipeline, or sequence")
        peers = []
        for value in range(sizes[dimension]):
            values = {
                "data": coordinate.data,
                "tensor": coordinate.tensor,
                "pipeline": coordinate.pipeline,
                "sequence": coordinate.sequence,
            }
            values[dimension] = value
            peers.append(self.rank_for_coordinate(RankCoordinate(**values)))
        return tuple(peers)

    def peer_groups(self, dimension: ParallelDimension) -> tuple[tuple[int, ...], ...]:
        """Partition the entire world into stable groups for one dimension."""

        groups: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for rank in range(self.world_size):
            group = self.peer_group(rank, dimension)
            if group not in seen:
                seen.add(group)
                groups.append(group)
        return tuple(groups)

    def to_mapping(self) -> dict[str, int]:
        """Return the canonical public topology representation."""

        return {
            "world_size": self.world_size,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel_size": self.sequence_parallel_size,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, int]) -> HybridParallelTopology:
        """Restore one topology from its exact canonical representation."""

        fields = {
            "world_size",
            "data_parallel_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "sequence_parallel_size",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("topology fields do not match the canonical schema")
        return cls(
            world_size=payload["world_size"],
            data_parallel_size=payload["data_parallel_size"],
            tensor_parallel_size=payload["tensor_parallel_size"],
            pipeline_parallel_size=payload["pipeline_parallel_size"],
            sequence_parallel_size=payload["sequence_parallel_size"],
        )
