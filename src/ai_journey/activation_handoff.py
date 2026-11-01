"""Canonical admission contracts for bounded pipeline activation handoffs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any

from .hybrid_topology import HybridParallelTopology, RankCoordinate
from .pipeline_partition import PipelinePartitionPlan, partition_transformer_layers


ACTIVATION_HANDOFF_SCHEMA = "ai-journey.activation-handoff.v1"
MAX_ACTIVATION_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_ACTIVATION_DIMENSIONS = 8
MAX_ACTIVATION_CONTRACT_BYTES = 4096

_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}
_CONTRACT_FIELDS = {
    "schema",
    "source_rank",
    "destination_rank",
    "tensor_shape",
    "dtype",
    "payload_length",
    "payload_sha256",
    "handoff_sha256",
}


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def activation_handoff_digest(
    *,
    source_rank: int,
    destination_rank: int,
    tensor_shape: Sequence[int],
    dtype: str,
    payload: bytes,
) -> str:
    """Bind exact activation bytes to their declared transfer metadata."""

    metadata = {
        "schema": ACTIVATION_HANDOFF_SCHEMA,
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "tensor_shape": list(tensor_shape),
        "dtype": dtype,
        "payload_length": len(payload),
    }
    return sha256(_canonical_json_bytes(metadata) + b"\x00" + payload).hexdigest()


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in pairs:
        if name in payload:
            raise ValueError(f"duplicate field: {name}")
        payload[name] = value
    return payload


def _validate_partition(
    topology: HybridParallelTopology, partition: PipelinePartitionPlan
) -> None:
    if not isinstance(partition, PipelinePartitionPlan):
        raise TypeError("partition must be a PipelinePartitionPlan")
    canonical = partition_transformer_layers(
        topology, layer_count=partition.layer_count
    )
    if partition != canonical:
        raise ValueError("partition must describe canonical stage ownership")


def _rank_coordinate(
    topology: HybridParallelTopology, rank: int, *, name: str
) -> RankCoordinate:
    if type(rank) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= rank < topology.world_size:
        raise ValueError(f"{name} is outside the topology")
    return topology.coordinate_for_rank(rank)


def _validate_route(
    topology: HybridParallelTopology, *, source_rank: int, destination_rank: int
) -> None:
    source = _rank_coordinate(topology, source_rank, name="source_rank")
    destination = _rank_coordinate(topology, destination_rank, name="destination_rank")
    if source.pipeline == destination.pipeline:
        raise ValueError(
            "source and destination must belong to different pipeline stages"
        )
    if (source.data, source.tensor, source.sequence) != (
        destination.data,
        destination.tensor,
        destination.sequence,
    ):
        raise ValueError("source and destination non-pipeline coordinates must match")
    if destination.pipeline != source.pipeline + 1:
        raise ValueError("destination must be the next adjacent pipeline stage")


def _validate_maximum_payload_bytes(maximum_payload_bytes: int) -> None:
    if type(maximum_payload_bytes) is not int:
        raise TypeError("maximum_payload_bytes must be an integer")
    if maximum_payload_bytes <= 0:
        raise ValueError("maximum_payload_bytes must be positive")
    if maximum_payload_bytes > MAX_ACTIVATION_PAYLOAD_BYTES:
        raise ValueError("maximum_payload_bytes cannot exceed system bound")


def _admit_shape(shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes, bytearray)) or not isinstance(shape, Sequence):
        raise TypeError("tensor_shape must be a sequence")
    admitted = tuple(shape)
    if not admitted:
        raise ValueError("tensor_shape must be non-empty")
    if len(admitted) > MAX_ACTIVATION_DIMENSIONS:
        raise ValueError("tensor_shape has too many dimensions")
    if any(type(size) is not int or size <= 0 for size in admitted):
        raise ValueError("tensor_shape dimensions must be positive integers")
    return admitted


@dataclass(frozen=True, init=False)
class ActivationHandoffContract:
    """The canonical metadata admitted before one activation transfer."""

    source_rank: int
    destination_rank: int
    tensor_shape: tuple[int, ...]
    dtype: str
    payload_length: int
    payload_sha256: str
    handoff_sha256: str

    @classmethod
    def _from_admitted_fields(
        cls,
        *,
        source_rank: int,
        destination_rank: int,
        tensor_shape: tuple[int, ...],
        dtype: str,
        payload_length: int,
        payload_sha256: str,
        handoff_sha256: str,
    ) -> ActivationHandoffContract:
        contract = object.__new__(cls)
        object.__setattr__(contract, "source_rank", source_rank)
        object.__setattr__(contract, "destination_rank", destination_rank)
        object.__setattr__(contract, "tensor_shape", tensor_shape)
        object.__setattr__(contract, "dtype", dtype)
        object.__setattr__(contract, "payload_length", payload_length)
        object.__setattr__(contract, "payload_sha256", payload_sha256)
        object.__setattr__(contract, "handoff_sha256", handoff_sha256)
        return contract

    @classmethod
    def admit(
        cls,
        *,
        topology: HybridParallelTopology,
        partition: PipelinePartitionPlan,
        source_rank: int,
        destination_rank: int,
        tensor_shape: Sequence[int],
        dtype: str,
        payload: bytes,
        maximum_payload_bytes: int = MAX_ACTIVATION_PAYLOAD_BYTES,
    ) -> ActivationHandoffContract:
        """Validate metadata and payload before allocation or communication."""

        if not isinstance(topology, HybridParallelTopology):
            raise TypeError("topology must be a HybridParallelTopology")
        _validate_partition(topology, partition)
        _validate_route(
            topology,
            source_rank=source_rank,
            destination_rank=destination_rank,
        )
        shape = _admit_shape(tensor_shape)
        if type(dtype) is not str or dtype not in _DTYPE_BYTES:
            raise ValueError("unsupported activation dtype")
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        _validate_maximum_payload_bytes(maximum_payload_bytes)
        expected_length = prod(shape) * _DTYPE_BYTES[dtype]
        if expected_length > maximum_payload_bytes:
            raise ValueError("activation exceeds the maximum payload bound")
        if len(payload) != expected_length:
            raise ValueError(
                f"activation payload must contain exactly {expected_length} bytes"
            )
        return cls._from_admitted_fields(
            source_rank=source_rank,
            destination_rank=destination_rank,
            tensor_shape=shape,
            dtype=dtype,
            payload_length=expected_length,
            payload_sha256=sha256(payload).hexdigest(),
            handoff_sha256=activation_handoff_digest(
                source_rank=source_rank,
                destination_rank=destination_rank,
                tensor_shape=shape,
                dtype=dtype,
                payload=payload,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact public schema in canonical field form."""

        return {
            "schema": ACTIVATION_HANDOFF_SCHEMA,
            "source_rank": self.source_rank,
            "destination_rank": self.destination_rank,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "payload_length": self.payload_length,
            "payload_sha256": self.payload_sha256,
            "handoff_sha256": self.handoff_sha256,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the contract into its unique canonical JSON encoding."""

        return _canonical_json_bytes(self.to_mapping())

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        *,
        topology: HybridParallelTopology,
        partition: PipelinePartitionPlan,
        payload: bytes,
        maximum_payload_bytes: int = MAX_ACTIVATION_PAYLOAD_BYTES,
    ) -> ActivationHandoffContract:
        """Admit one exact mapping and recompute all payload-derived fields."""

        if not isinstance(mapping, Mapping) or set(mapping) != _CONTRACT_FIELDS:
            raise ValueError(
                "activation handoff fields do not match the canonical schema"
            )
        if mapping["schema"] != ACTIVATION_HANDOFF_SCHEMA:
            raise ValueError("activation handoff schema is unsupported")
        shape = mapping["tensor_shape"]
        if not isinstance(shape, list):
            raise ValueError("tensor_shape must use its canonical list representation")
        admitted = cls.admit(
            topology=topology,
            partition=partition,
            source_rank=mapping["source_rank"],  # type: ignore[arg-type]
            destination_rank=mapping["destination_rank"],  # type: ignore[arg-type]
            tensor_shape=shape,
            dtype=mapping["dtype"],  # type: ignore[arg-type]
            payload=payload,
            maximum_payload_bytes=maximum_payload_bytes,
        )
        if mapping["payload_length"] != admitted.payload_length:
            raise ValueError("declared payload length differs from tensor metadata")
        if mapping["payload_sha256"] != admitted.payload_sha256:
            raise ValueError("payload digest differs")
        if mapping["handoff_sha256"] != admitted.handoff_sha256:
            raise ValueError("handoff digest differs")
        return admitted

    @classmethod
    def from_json_bytes(
        cls,
        raw: bytes,
        *,
        topology: HybridParallelTopology,
        partition: PipelinePartitionPlan,
        payload: bytes,
        maximum_payload_bytes: int = MAX_ACTIVATION_PAYLOAD_BYTES,
    ) -> ActivationHandoffContract:
        """Parse duplicate-safe canonical JSON and admit its bound payload."""

        if not isinstance(raw, bytes):
            raise TypeError("contract JSON must be bytes")
        if len(raw) > MAX_ACTIVATION_CONTRACT_BYTES:
            raise ValueError("activation handoff contract exceeds its byte bound")
        try:
            mapping = json.loads(raw, object_pairs_hook=_reject_duplicate_fields)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("activation handoff contract is not valid JSON") from error
        if not isinstance(mapping, dict):
            raise ValueError("activation handoff contract must be a JSON object")
        if raw != _canonical_json_bytes(mapping):
            raise ValueError("activation handoff contract is not canonical JSON")
        return cls.from_mapping(
            mapping,
            topology=topology,
            partition=partition,
            payload=payload,
            maximum_payload_bytes=maximum_payload_bytes,
        )
