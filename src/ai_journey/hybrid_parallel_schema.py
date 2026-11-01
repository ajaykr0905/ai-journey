"""Canonical schema and bounded admission for hybrid-parallel reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hybrid_topology import HybridParallelTopology
from .parallel_communication import CommunicationAssumptions
from .pipeline_handoff_probe import PipelineHandoffProbeResult
from .tensor_sharding import LogicalTensor, MAX_TENSOR_DIMENSIONS, MAX_TENSOR_ELEMENTS


REPORT_SCHEMA = "ai-journey.hybrid-parallel-report.v1"
REPORT_KIND = "hybrid_parallel_planning_and_cpu_gloo_conformance"
MAX_REPORT_WORLD_SIZE = 4_096
MAX_REPORT_LAYER_COUNT = 4_096
MAX_REPORT_MICROBATCH_COUNT = 4_096
MAX_REPORT_SCHEDULE_OPERATIONS = 65_536
MAX_REPORT_SEQUENCE_TOKENS = 1_000_000
EVIDENCE_BOUNDARY = (
    "Topology, sharding, partition, schedule, Pipeline Bubble, and communication "
    "results are deterministic analytical evidence. The report records one bounded "
    "local CPU/Gloo activation-handoff observation. Its SHA-256 seal verifies report "
    "integrity and replayed internal consistency; it does not independently "
    "authenticate execution. This report does not establish NVIDIA GPU, NCCL, "
    "accelerator, multi-node, performance, throughput, utilization, memory, cost, "
    "or production-readiness evidence."
)
TOP_LEVEL_FIELDS = {
    "schema",
    "kind",
    "inputs",
    "analytical",
    "executed_observations",
    "claims",
    "evidence_boundary",
    "integrity",
}
INPUT_FIELDS = {
    "topology",
    "tensor",
    "sequence",
    "pipeline",
    "communication_assumptions",
    "probe_observation",
}


class HybridParallelReportError(ValueError):
    """Raised when hybrid-parallel evidence cannot be admitted or replayed."""


def exact_mapping(value: object, fields: set[str], *, name: str) -> Mapping[str, Any]:
    """Admit an object with exactly the declared fields."""

    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields do not match the canonical schema")
    return value


def _bounded_positive(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{name} exceeds the report admission bound")
    return value


def admit_generation_bounds(
    *,
    topology: HybridParallelTopology,
    tensor: LogicalTensor,
    token_ids: tuple[int, ...],
    layer_count: int,
    microbatch_count: int,
) -> None:
    """Apply every report expansion cap before analytical materialization."""

    _bounded_positive(
        topology.world_size, name="world_size", maximum=MAX_REPORT_WORLD_SIZE
    )
    if len(tensor.shape) > MAX_TENSOR_DIMENSIONS:
        raise ValueError("tensor shape exceeds the report admission bound")
    if len(tensor.values) > MAX_TENSOR_ELEMENTS:
        raise ValueError("tensor values exceed the report admission bound")
    if len(token_ids) > MAX_REPORT_SEQUENCE_TOKENS:
        raise ValueError("token_ids exceeds the report admission bound")
    _bounded_positive(layer_count, name="layer_count", maximum=MAX_REPORT_LAYER_COUNT)
    _bounded_positive(
        microbatch_count,
        name="microbatch_count",
        maximum=MAX_REPORT_MICROBATCH_COUNT,
    )
    operation_count = 2 * topology.pipeline_parallel_size * microbatch_count
    if operation_count > MAX_REPORT_SCHEDULE_OPERATIONS:
        raise ValueError("pipeline schedule exceeds the report admission bound")


def admit_report_inputs(
    value: object,
) -> tuple[
    HybridParallelTopology,
    LogicalTensor,
    tuple[int, ...],
    int,
    str,
    int,
    CommunicationAssumptions,
    dict[str, object],
]:
    """Admit primary report inputs before rebuilding any derived collection."""

    inputs = exact_mapping(value, INPUT_FIELDS, name="report inputs")
    topology_payload = exact_mapping(
        inputs["topology"],
        {
            "world_size",
            "data_parallel_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "sequence_parallel_size",
        },
        name="topology",
    )
    _bounded_positive(
        topology_payload["world_size"],
        name="world_size",
        maximum=MAX_REPORT_WORLD_SIZE,
    )
    for name in (
        "data_parallel_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "sequence_parallel_size",
    ):
        _bounded_positive(
            topology_payload[name], name=name, maximum=MAX_REPORT_WORLD_SIZE
        )

    tensor_payload = exact_mapping(
        inputs["tensor"], {"shape", "values"}, name="tensor input"
    )
    sequence_payload = exact_mapping(
        inputs["sequence"], {"token_ids"}, name="sequence input"
    )
    pipeline_payload = exact_mapping(
        inputs["pipeline"],
        {"layer_count", "schedule_policy", "microbatch_count"},
        name="pipeline input",
    )
    assumptions_payload = exact_mapping(
        inputs["communication_assumptions"],
        {"element_size_bytes", "pipeline_payload_bytes"},
        name="communication assumptions",
    )
    shape = tensor_payload["shape"]
    values = tensor_payload["values"]
    token_ids = sequence_payload["token_ids"]
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > MAX_TENSOR_DIMENSIONS
        or not isinstance(values, list)
        or len(values) > MAX_TENSOR_ELEMENTS
    ):
        raise ValueError("tensor inputs exceed the report admission bound")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) > MAX_REPORT_SEQUENCE_TOKENS
        or any(type(token) is not int for token in token_ids)
    ):
        raise ValueError("sequence inputs exceed the report admission bound")
    layer_count = _bounded_positive(
        pipeline_payload["layer_count"],
        name="layer_count",
        maximum=MAX_REPORT_LAYER_COUNT,
    )
    microbatch_count = _bounded_positive(
        pipeline_payload["microbatch_count"],
        name="microbatch_count",
        maximum=MAX_REPORT_MICROBATCH_COUNT,
    )
    schedule_policy = pipeline_payload["schedule_policy"]
    if schedule_policy not in ("gpipe", "1f1b"):
        raise ValueError("schedule_policy must be gpipe or 1f1b")

    topology = HybridParallelTopology.from_mapping(topology_payload)
    if 2 * topology.pipeline_parallel_size * microbatch_count > (
        MAX_REPORT_SCHEDULE_OPERATIONS
    ):
        raise ValueError("pipeline schedule exceeds the report admission bound")
    tensor = LogicalTensor(tuple(shape), tuple(values))
    admitted_tokens = tuple(token_ids)
    assumptions = CommunicationAssumptions(**assumptions_payload)
    probe = PipelineHandoffProbeResult.admit_mapping(
        inputs["probe_observation"],
        topology=topology,
        layer_count=layer_count,
    )
    return (
        topology,
        tensor,
        admitted_tokens,
        layer_count,
        schedule_policy,
        microbatch_count,
        assumptions,
        probe,
    )
