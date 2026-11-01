"""Analytical communication accounting for admitted hybrid-parallel plans.

The public results describe declared algorithms and aggregate modeled wire
traffic. They do not execute communication or infer measured performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .hybrid_topology import HybridParallelTopology
from .pipeline_schedule import PipelineSchedule, validate_pipeline_schedule
from .sequence_sharding import SequenceShard, reconstruct_sequence_shards
from .tensor_sharding import TensorShard, reconstruct_tensor


MAX_ACCOUNTED_BYTES = (1 << 63) - 1
TENSOR_COLLECTIVE = "ring_all_gather_equal_ceiling_chunks"
SEQUENCE_TRANSFER = "direct_all_to_all_equal_ceiling_chunks"
PIPELINE_HANDOFF = "point_to_point_cross_stage_dependencies"
WIRE_SCOPE = "aggregate_across_participating_ranks"
EVIDENCE_BOUNDARY = (
    "Analytical communication accounting from admitted plans and explicit wire "
    "assumptions; it does not execute collectives, measure bandwidth, or establish "
    "accelerator, NCCL, multi-node, performance, or production evidence."
)


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > MAX_ACCOUNTED_BYTES:
        raise ValueError(f"{name} exceeds the accounting byte bound")
    return value


def _bounded_product(*values: int) -> int:
    result = math.prod(values)
    if result > MAX_ACCOUNTED_BYTES:
        raise ValueError("communication byte calculation exceeds the accounting bound")
    return result


@dataclass(frozen=True)
class CommunicationAssumptions:
    """Explicit byte widths and supported analytical wire algorithms."""

    element_size_bytes: int
    pipeline_payload_bytes: int
    tensor_collective: str = TENSOR_COLLECTIVE
    sequence_transfer: str = SEQUENCE_TRANSFER
    pipeline_handoff: str = PIPELINE_HANDOFF
    wire_scope: str = WIRE_SCOPE

    def __post_init__(self) -> None:
        _positive_integer(self.element_size_bytes, "element_size_bytes")
        _positive_integer(self.pipeline_payload_bytes, "pipeline_payload_bytes")
        if self.tensor_collective != TENSOR_COLLECTIVE:
            raise ValueError("unsupported tensor collective assumption")
        if self.sequence_transfer != SEQUENCE_TRANSFER:
            raise ValueError("unsupported sequence transfer assumption")
        if self.pipeline_handoff != PIPELINE_HANDOFF:
            raise ValueError("unsupported pipeline handoff assumption")
        if self.wire_scope != WIRE_SCOPE:
            raise ValueError("unsupported wire scope assumption")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical public assumption representation."""

        return {
            "element_size_bytes": self.element_size_bytes,
            "pipeline_payload_bytes": self.pipeline_payload_bytes,
            "tensor_collective": self.tensor_collective,
            "sequence_transfer": self.sequence_transfer,
            "pipeline_handoff": self.pipeline_handoff,
            "wire_scope": self.wire_scope,
        }


@dataclass(frozen=True)
class CommunicationVolume:
    """One dimension's logical payload and assumed aggregate wire work."""

    dimension: str
    operation: str
    operation_count: int
    payload_bytes_per_operation: int
    logical_bytes: int
    wire_bytes: int

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic values suitable for a Sealed Report."""

        return {
            "dimension": self.dimension,
            "operation": self.operation,
            "operation_count": self.operation_count,
            "payload_bytes_per_operation": self.payload_bytes_per_operation,
            "logical_bytes": self.logical_bytes,
            "wire_bytes": self.wire_bytes,
        }


@dataclass(frozen=True)
class ParallelCommunicationAccounting:
    """Separate tensor, sequence, and pipeline analytical communication."""

    assumptions: CommunicationAssumptions
    tensor_collective: CommunicationVolume
    sequence_transfer: CommunicationVolume
    pipeline_handoff: CommunicationVolume
    evidence_boundary: str = EVIDENCE_BOUNDARY

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical public accounting representation."""

        return {
            "assumptions": self.assumptions.to_mapping(),
            "tensor_collective": self.tensor_collective.to_mapping(),
            "sequence_transfer": self.sequence_transfer.to_mapping(),
            "pipeline_handoff": self.pipeline_handoff.to_mapping(),
            "evidence_boundary": self.evidence_boundary,
        }


def account_parallel_communication(
    *,
    topology: HybridParallelTopology,
    tensor_shards: tuple[TensorShard, ...],
    sequence_shards: tuple[SequenceShard, ...],
    schedule: PipelineSchedule,
    assumptions: CommunicationAssumptions,
) -> ParallelCommunicationAccounting:
    """Account communication only after all referenced plans are admitted."""

    if not isinstance(topology, HybridParallelTopology):
        raise TypeError("topology must be a HybridParallelTopology")
    if not isinstance(assumptions, CommunicationAssumptions):
        raise TypeError("assumptions must be CommunicationAssumptions")
    if not isinstance(schedule, PipelineSchedule):
        raise TypeError("schedule must be a PipelineSchedule")

    tensor = reconstruct_tensor(tensor_shards)
    if len(tensor_shards) != topology.tensor_parallel_size:
        raise ValueError("tensor shards are incompatible with the topology")
    sequence = reconstruct_sequence_shards(sequence_shards, topology)
    validate_pipeline_schedule(schedule)
    if schedule.stage_count != topology.pipeline_parallel_size:
        raise ValueError("pipeline schedule is incompatible with the topology")

    tensor_size = topology.tensor_parallel_size
    tensor_shard_elements = tensor_shards[0].padded_length
    tensor_payload = _bounded_product(
        tensor_shard_elements, tensor_size, assumptions.element_size_bytes
    )
    tensor_logical = _bounded_product(
        len(tensor.values), assumptions.element_size_bytes
    )
    tensor_wire = _bounded_product(
        tensor_size,
        tensor_size - 1,
        tensor_shard_elements,
        assumptions.element_size_bytes,
    )

    sequence_size = topology.sequence_parallel_size
    sequence_width = sequence_shards[0].padded_length // sequence_size
    sequence_segment_elements = math.ceil(sequence_width / sequence_size)
    sequence_payload = _bounded_product(
        sequence_shards[0].padded_length, assumptions.element_size_bytes
    )
    sequence_logical = _bounded_product(len(sequence), assumptions.element_size_bytes)
    sequence_wire = _bounded_product(
        sequence_size,
        sequence_size - 1,
        sequence_segment_elements,
        assumptions.element_size_bytes,
    )

    handoff_count = sum(
        operation.stage_index > 0
        if operation.direction == "forward"
        else operation.stage_index < schedule.stage_count - 1
        for slot in schedule.slots
        for operation in slot.operations
    )
    pipeline_logical = _bounded_product(
        handoff_count, assumptions.pipeline_payload_bytes
    )

    return ParallelCommunicationAccounting(
        assumptions=assumptions,
        tensor_collective=CommunicationVolume(
            "tensor", "ring_all_gather", 1, tensor_payload, tensor_logical, tensor_wire
        ),
        sequence_transfer=CommunicationVolume(
            "sequence",
            "all_to_all",
            1,
            sequence_payload,
            sequence_logical,
            sequence_wire,
        ),
        pipeline_handoff=CommunicationVolume(
            "pipeline",
            "point_to_point",
            handoff_count,
            assumptions.pipeline_payload_bytes,
            pipeline_logical,
            pipeline_logical,
        ),
    )
