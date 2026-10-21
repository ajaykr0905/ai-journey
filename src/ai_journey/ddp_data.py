"""Deterministic rank-local views of one public byte stream."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

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


@dataclass(frozen=True)
class BatchPlan:
    sample_count: int
    world_size: int
    micro_batch_size: int
    accumulation_steps: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            integer(getattr(self, name), name, 1)
        if self.updates_per_epoch == 0:
            raise ValueError("corpus cannot supply one complete global update")

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.micro_batch_size * self.accumulation_steps

    @property
    def updates_per_epoch(self) -> int:
        return self.sample_count // self.global_batch_size

    @property
    def discarded_samples(self) -> int:
        return self.sample_count % self.global_batch_size


def byte_batches(
    source: bytes,
    *,
    sequence_length: int,
    topology: Topology,
    micro_batch_size: int,
    accumulation_steps: int,
    seed: int,
    epoch: int = 0,
):
    """Build equal complete updates from next-byte samples."""
    integer(sequence_length, "sequence_length", 1)
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    count = (len(source) - 1) // sequence_length
    plan = BatchPlan(count, topology.world_size, micro_batch_size, accumulation_steps)
    usable = count - plan.discarded_samples
    ids = shard_indices(usable, topology, seed=seed, epoch=epoch)
    tokens = torch.tensor(list(source), dtype=torch.long)
    result = []
    for start in range(0, len(ids), micro_batch_size):
        offsets = [
            sample * sequence_length for sample in ids[start : start + micro_batch_size]
        ]
        inputs = torch.stack(
            [tokens[offset : offset + sequence_length] for offset in offsets]
        )
        targets = torch.stack(
            [tokens[offset + 1 : offset + sequence_length + 1] for offset in offsets]
        )
        result.append((inputs, targets))
    return tuple(result), plan
