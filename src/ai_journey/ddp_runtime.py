"""Owned Gloo process groups with bounded collective timeouts."""

from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
import json

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
    if tokens == 0 and loss_sum == 0:
        pass
    else:
        weighted_mean([(loss_sum, tokens)])
    values = torch.tensor([loss_sum, tokens], dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_loss, total_tokens = values.tolist()
    if total_tokens == 0:
        raise ValueError("global evaluation contains no targets")
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


def require_optimizer_consensus(optimizer) -> str:
    """Compare tensor bytes and all optimizer settings across replicas."""

    def encode(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return [
                "tensor",
                str(tensor.dtype),
                list(tensor.shape),
                sha256(
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            ]
        if isinstance(value, dict):
            return [
                "dict",
                [
                    [encode(key), encode(item)]
                    for key, item in sorted(
                        value.items(),
                        key=lambda pair: (type(pair[0]).__name__, repr(pair[0])),
                    )
                ],
            ]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(item) for item in value]]
        if value is None or type(value) in (bool, int, float, str):
            return [type(value).__name__, value]
        raise ValueError("unsupported optimizer state value")

    fingerprint = None
    try:
        fingerprint = sha256(
            json.dumps(encode(optimizer.state_dict()), allow_nan=False).encode()
        ).hexdigest()
    except (TypeError, ValueError, RuntimeError):
        pass
    require_all_ranks(fingerprint is not None, phase="optimizer state fingerprinting")
    encoded = torch.tensor(list(bytes.fromhex(fingerprint)), dtype=torch.uint8)
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    if any(not torch.equal(encoded, value) for value in gathered):
        raise RuntimeError("replicated optimizer fingerprints diverged")
    return fingerprint
