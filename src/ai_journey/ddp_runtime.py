"""Owned Gloo process groups with bounded collective timeouts."""

from contextlib import contextmanager
from datetime import timedelta

import torch.distributed as dist
import torch

from .ddp_metrics import weighted_mean
from .gpt2_training import gpt2_model_fingerprint

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


def reduce_token_totals(loss_sum: float, tokens: int) -> tuple[float, int, float]:
    """Reduce additive statistics; never average rank means."""
    weighted_mean([(loss_sum, tokens)])
    values = torch.tensor([loss_sum, tokens], dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_loss, total_tokens = values.tolist()
    return total_loss, int(total_tokens), total_loss / total_tokens


def require_all_ranks(success: bool, *, phase: str) -> None:
    """Make rank-local validation failure a common exception before the next phase."""
    if not isinstance(success, bool):
        raise TypeError("success must be boolean")
    if not isinstance(phase, str) or not phase or len(phase) > 120:
        raise ValueError("phase must be a bounded label")
    status = torch.tensor(int(success), dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if status.item() == 0:
        raise RuntimeError(f"at least one rank failed {phase}")


def require_model_consensus(model) -> str:
    """Prove replicated model bytes agree before checkpointing or serving."""
    fingerprint = gpt2_model_fingerprint(model)
    encoded = torch.tensor(list(bytes.fromhex(fingerprint)), dtype=torch.uint8)
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    if any(not torch.equal(encoded, value) for value in gathered):
        raise RuntimeError("replicated model fingerprints diverged")
    return fingerprint
