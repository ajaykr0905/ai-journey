"""Streaming causal averages for bounded-memory inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch
from torch import Tensor

from .causal_average import CausalAverageError, validate_values


@dataclass(frozen=True)
class StreamSnapshot:
    """Complete restart state for a causal-average stream."""

    count: int
    total: Tensor


SNAPSHOT_SCHEMA = "ai-journey-causal-stream-v1"


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest()


def build_snapshot_payload(snapshot: StreamSnapshot) -> dict[str, Any]:
    """Serialize CPU stream state into canonical, content-addressed data."""

    if not isinstance(snapshot, StreamSnapshot):
        raise TypeError("snapshot must be StreamSnapshot")
    if snapshot.total.device.type != "cpu":
        raise CausalAverageError("serialized stream snapshots must be on CPU")
    if snapshot.total.dtype not in (torch.float32, torch.float64):
        raise CausalAverageError(
            "serialized stream snapshots require float32 or float64"
        )
    if snapshot.total.ndim != 1 or not torch.isfinite(snapshot.total).all():
        raise CausalAverageError("snapshot total must be a finite vector")
    if (
        isinstance(snapshot.count, bool)
        or not isinstance(snapshot.count, int)
        or snapshot.count < 0
    ):
        raise CausalAverageError("snapshot count must be a non-negative integer")
    core: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "count": snapshot.count,
        "dtype": str(snapshot.total.dtype).removeprefix("torch."),
        "total": snapshot.total.tolist(),
    }
    return {**core, "sha256": _fingerprint(core)}


def parse_snapshot_payload(payload: dict[str, Any]) -> StreamSnapshot:
    """Verify and deserialize a canonical stream snapshot payload."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    required = {"schema", "count", "dtype", "total", "sha256"}
    if set(payload) != required:
        raise CausalAverageError("snapshot payload fields are invalid")
    core = {key: payload[key] for key in required - {"sha256"}}
    if payload["schema"] != SNAPSHOT_SCHEMA:
        raise CausalAverageError("snapshot schema is unsupported")
    if payload["sha256"] != _fingerprint(core):
        raise CausalAverageError("snapshot fingerprint mismatch")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(payload["dtype"])
    if dtype is None:
        raise CausalAverageError("snapshot dtype is unsupported")
    try:
        total = torch.tensor(payload["total"], dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise CausalAverageError("snapshot total is invalid") from exc
    snapshot = StreamSnapshot(payload["count"], total)
    build_snapshot_payload(snapshot)
    return snapshot


class CausalAverageStream:
    """Maintain prefix count and sum while processing consecutive chunks."""

    def __init__(
        self,
        feature_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(feature_size, bool) or not isinstance(feature_size, int):
            raise TypeError("feature_size must be an integer")
        if feature_size <= 0:
            raise CausalAverageError("feature_size must be positive")
        if not dtype.is_floating_point:
            raise CausalAverageError("stream dtype must be floating-point")
        self._sum = torch.zeros(feature_size, dtype=dtype, device=device)
        self._count = 0

    @property
    def count(self) -> int:
        """Number of valid positions consumed so far."""

        return self._count

    def update(self, chunk: Tensor) -> Tensor:
        """Return causal averages for one non-empty consecutive chunk."""

        validate_values(chunk)
        if chunk.ndim != 2:
            raise CausalAverageError("stream chunks must have shape (time, channels)")
        if chunk.shape[1] != self._sum.numel():
            raise CausalAverageError("chunk feature size does not match stream state")
        if chunk.dtype != self._sum.dtype or chunk.device != self._sum.device:
            raise CausalAverageError("chunk dtype and device must match stream state")
        cumulative = chunk.cumsum(dim=0) + self._sum
        counts = torch.arange(
            self._count + 1,
            self._count + len(chunk) + 1,
            dtype=chunk.dtype,
            device=chunk.device,
        ).unsqueeze(-1)
        output = cumulative / counts
        self._sum = cumulative[-1].detach().clone()
        self._count += len(chunk)
        return output

    def snapshot(self) -> StreamSnapshot:
        """Return a storage-independent copy of the complete stream state."""

        return StreamSnapshot(self._count, self._sum.detach().clone())

    def restore(self, snapshot: StreamSnapshot) -> None:
        """Validate and transactionally restore a prior snapshot."""

        if not isinstance(snapshot, StreamSnapshot):
            raise TypeError("snapshot must be StreamSnapshot")
        if (
            isinstance(snapshot.count, bool)
            or not isinstance(snapshot.count, int)
            or snapshot.count < 0
        ):
            raise CausalAverageError("snapshot count must be a non-negative integer")
        total = snapshot.total
        if not isinstance(total, Tensor):
            raise TypeError("snapshot total must be a torch.Tensor")
        if total.shape != self._sum.shape:
            raise CausalAverageError(
                "snapshot feature size does not match stream state"
            )
        if total.dtype != self._sum.dtype or total.device != self._sum.device:
            raise CausalAverageError(
                "snapshot dtype and device must match stream state"
            )
        if not torch.isfinite(total).all():
            raise CausalAverageError("snapshot total must be finite")
        self._sum = total.detach().clone()
        self._count = snapshot.count
