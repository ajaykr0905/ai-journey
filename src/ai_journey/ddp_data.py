"""Deterministic rank-local views of one public byte stream."""

from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha256
import json
from types import SimpleNamespace

import torch

from .ddp_contracts import Topology, integer


SAMPLER_V1 = "rank-strided-truncated-shuffle-v1"
SAMPLER_V2 = "rank-strided-full-epoch-shuffle-v2"
SAMPLERS = (SAMPLER_V1, SAMPLER_V2)


def validate_sampler(sampler):
    if type(sampler) is not str or sampler not in SAMPLERS:
        raise ValueError("unsupported dense DDP sampler")
    return sampler


def sampler_options(sampler):
    """Keep omitted/v1 caller contracts unchanged, including legacy adapters."""
    return {} if validate_sampler(sampler) == SAMPLER_V1 else {"sampler": sampler}


def data_contract(source: bytes, config, world_size: int) -> dict:
    """Canonical v2 replay identity, scoped to dense rank-local byte sampling."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    return _data_contract(config, world_size, sha256(source).hexdigest(), len(source))


def _data_contract(config, world_size, source_sha256, source_bytes):
    integer(world_size, "world_size", 1)
    integer(source_bytes, "source_bytes", 2)
    if world_size > 4 or source_bytes > 1024 * 1024:
        raise ValueError("data contract exceeds the CPU topology or source budget")
    if (
        type(source_sha256) is not str
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("data contract source identity must be SHA-256")
    for name in ("sequence_length", "micro_batch_size", "accumulation_steps"):
        integer(getattr(config, name), name, 1)
    integer(config.seed, "seed")
    BatchPlan(
        (source_bytes - 1) // config.sequence_length,
        world_size,
        config.micro_batch_size,
        config.accumulation_steps,
    )
    return {
        "schema": "ai-journey-ddp-data-contract-v2",
        "sampler": SAMPLER_V2,
        "source_sha256": source_sha256,
        "source_bytes": source_bytes,
        "world_size": world_size,
        "sequence_length": config.sequence_length,
        "micro_batch_size": config.micro_batch_size,
        "accumulation_steps": config.accumulation_steps,
        "seed": config.seed,
    }


def validate_data_contract(contract, config=None, *, world_size, source_sha256):
    """Reject extra fields and scalar coercions, including bool/int aliases."""
    if type(contract) is not dict or "source_bytes" not in contract:
        raise ValueError("dense sampler data contract must be a mapping")
    if config is None:
        required = ("sequence_length", "micro_batch_size", "accumulation_steps", "seed")
        if any(name not in contract for name in required):
            raise ValueError("dense sampler data contract is incomplete")
        config = SimpleNamespace(**{name: contract[name] for name in required})
    expected = _data_contract(
        config, world_size, source_sha256, contract["source_bytes"]
    )
    if json.dumps(contract, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("dense sampler data contract does not match")
    return contract


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
    sampler: str = SAMPLER_V1,
):
    """Build equal complete updates from next-byte samples."""
    integer(sequence_length, "sequence_length", 1)
    validate_sampler(sampler)
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    count = (len(source) - 1) // sequence_length
    plan = BatchPlan(count, topology.world_size, micro_batch_size, accumulation_steps)
    usable = count - plan.discarded_samples
    ids = shard_indices(
        count if sampler == SAMPLER_V2 else usable,
        topology,
        seed=seed,
        epoch=epoch,
    )[: usable // topology.world_size]
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
