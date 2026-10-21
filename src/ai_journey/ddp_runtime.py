"""Owned Gloo process groups with bounded collective timeouts."""

from contextlib import contextmanager
from datetime import timedelta

import torch.distributed as dist

from .ddp_contracts import Topology, integer


@contextmanager
def process_group(topology: Topology, *, rendezvous: str, timeout_seconds: int = 30):
    integer(timeout_seconds, "timeout_seconds", 1)
    if not isinstance(topology, Topology):
        raise TypeError("topology must be Topology")
    if not isinstance(rendezvous, str) or not rendezvous.startswith(
        ("file://", "tcp://", "env://")
    ):
        raise ValueError("unsupported rendezvous URL")
    if dist.is_initialized():
        raise RuntimeError("a caller-owned process group already exists")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("the CPU distributed lab requires Gloo")
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=topology.rank,
        world_size=topology.world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        yield topology
    finally:
        dist.destroy_process_group()
