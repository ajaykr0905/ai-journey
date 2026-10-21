"""Deterministic rank-local views of one public byte stream."""

from __future__ import annotations

import random

from .ddp_contracts import Topology, integer


def shard_indices(
    sample_count: int, topology: Topology, *, seed: int, epoch: int = 0
) -> tuple[int, ...]:
    """Shuffle once globally, drop only the tail, then partition without overlap."""
    integer(sample_count, "sample_count", 1)
    integer(seed, "seed")
    integer(epoch, "epoch")
    if not isinstance(topology, Topology):
        raise TypeError("topology must be Topology")
    order = list(range(sample_count))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    usable = sample_count - sample_count % topology.world_size
    return tuple(order[topology.rank : usable : topology.world_size])
