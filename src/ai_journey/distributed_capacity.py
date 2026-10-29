"""Analytical distributed-training capacity primitives.

These helpers model declared topology and link assumptions. They do not probe
hardware, execute collectives, or claim measured network performance.
"""

from __future__ import annotations

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
