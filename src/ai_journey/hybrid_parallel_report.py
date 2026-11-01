"""Replayable Sealed Reports for bounded hybrid-parallelism evidence."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
from typing import Any

from .activation_handoff import ACTIVATION_HANDOFF_SCHEMA
from .hybrid_topology import HybridParallelTopology
from .parallel_communication import (
    CommunicationAssumptions,
    account_parallel_communication,
)
from .pipeline_handoff_probe import PROBE_SCHEMA, PipelineHandoffProbeResult
from .pipeline_metrics import measure_pipeline_bubbles
from .pipeline_partition import partition_transformer_layers
from .pipeline_schedule import generate_1f1b_schedule, generate_gpipe_schedule
from .sequence_sharding import plan_sequence_shards, reconstruct_sequence_shards
from .tensor_sharding import LogicalTensor, plan_tensor_shards, reconstruct_tensor


REPORT_SCHEMA = "ai-journey.hybrid-parallel-report.v1"
REPORT_KIND = "hybrid_parallel_planning_and_cpu_gloo_conformance"
EVIDENCE_BOUNDARY = (
    "Topology, sharding, partition, schedule, Pipeline Bubble, and communication "
    "results are deterministic analytical evidence. The executed observation is "
    "limited to one bounded local CPU/Gloo activation handoff. This report does "
    "not establish NVIDIA GPU, NCCL, accelerator, multi-node, performance, "
    "throughput, utilization, memory, cost, or production-readiness evidence."
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "kind",
    "inputs",
    "analytical",
    "executed_observations",
    "claims",
    "evidence_boundary",
    "integrity",
}
_INPUT_FIELDS = {
    "topology",
    "tensor",
    "sequence",
    "pipeline",
    "communication_assumptions",
    "probe_observation",
}
_PROBE_FIELDS = {
    "schema",
    "executed",
    "performance_measured",
    "backend",
    "device",
    "world_size",
    "outcome",
    "contract",
    "rank_outcomes",
    "limitations",
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
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
_PROBE_LIMITATIONS = [
    "CPU/Gloo transfer correctness only; no NVIDIA GPU or NCCL was used.",
    "No latency, bandwidth, throughput, utilization, memory, or cost was measured.",
]
_RANK_OUTCOMES = [
    {"probe_rank": 0, "role": "source", "outcome": "success"},
    {"probe_rank": 1, "role": "destination", "outcome": "success"},
]


class HybridParallelReportError(ValueError):
    """Raised when hybrid-parallel evidence cannot be admitted or replayed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hybrid_parallel_report_bytes(report: Mapping[str, object]) -> bytes:
    """Return the report's unique UTF-8 JSON representation."""

    if not isinstance(report, Mapping):
        raise TypeError("hybrid parallel report must be a mapping")
    try:
        return _canonical_json_bytes(report)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report cannot be canonically encoded"
        ) from error


def _unsigned(report: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in report.items() if key != "integrity"}


def seal_hybrid_parallel_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Attach a recomputable SHA-256 seal without mutating the input."""

    if not isinstance(report, Mapping):
        raise TypeError("hybrid parallel report must be a mapping")
    unsigned = _unsigned(report)
    try:
        digest = sha256(_canonical_json_bytes(unsigned)).hexdigest()
        sealed = json.loads(_canonical_json_bytes(unsigned))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report cannot be canonically encoded"
        ) from error
    sealed["integrity"] = {"algorithm": "sha256", "digest": digest}
    return sealed


def _exact_mapping(value: object, fields: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields do not match the canonical schema")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_probe_observation(
    value: object,
    *,
    topology: HybridParallelTopology,
    layer_count: int,
) -> dict[str, object]:
    try:
        probe = _exact_mapping(value, _PROBE_FIELDS, name="probe observation")
        fixed = {
            "schema": PROBE_SCHEMA,
            "executed": True,
            "performance_measured": False,
            "backend": "gloo",
            "device": "cpu",
            "world_size": 2,
            "outcome": "success",
            "rank_outcomes": _RANK_OUTCOMES,
            "limitations": _PROBE_LIMITATIONS,
        }
        if any(probe.get(key) != expected for key, expected in fixed.items()):
            raise ValueError("probe observation escalates or contradicts its evidence")
        contract = _exact_mapping(
            probe["contract"], _CONTRACT_FIELDS, name="probe contract"
        )
        if contract["schema"] != ACTIVATION_HANDOFF_SCHEMA:
            raise ValueError("probe contract schema is unsupported")
        source_rank = contract["source_rank"]
        destination_rank = contract["destination_rank"]
        if type(source_rank) is not int or type(destination_rank) is not int:
            raise TypeError("probe contract ranks must be integers")
        source = topology.coordinate_for_rank(source_rank)
        destination = topology.coordinate_for_rank(destination_rank)
        if destination.pipeline != source.pipeline + 1 or (
            source.data,
            source.tensor,
            source.sequence,
        ) != (destination.data, destination.tensor, destination.sequence):
            raise ValueError("probe contract route is outside adjacent pipeline stages")
        partition_transformer_layers(topology, layer_count=layer_count)
        shape = contract["tensor_shape"]
        dtype = contract["dtype"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
            or dtype not in _DTYPE_BYTES
        ):
            raise ValueError("probe contract tensor metadata is invalid")
        expected_length = math.prod(shape) * _DTYPE_BYTES[dtype]
        if contract["payload_length"] != expected_length:
            raise ValueError("probe contract payload length is inconsistent")
        _digest(contract["payload_sha256"], name="probe payload digest")
        _digest(contract["handoff_sha256"], name="probe handoff digest")
        return json.loads(_canonical_json_bytes(probe))
    except HybridParallelReportError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise HybridParallelReportError("invalid CPU/Gloo probe observation") from error


def _topology_analysis(topology: HybridParallelTopology) -> dict[str, object]:
    coordinates = []
    for rank in range(topology.world_size):
        coordinate = topology.coordinate_for_rank(rank)
        coordinates.append(
            {
                "rank": rank,
                "data": coordinate.data,
                "tensor": coordinate.tensor,
                "pipeline": coordinate.pipeline,
                "sequence": coordinate.sequence,
            }
        )
    return {
        "coordinates": coordinates,
        "peer_groups": {
            dimension: [list(group) for group in topology.peer_groups(dimension)]
            for dimension in ("data", "tensor", "pipeline", "sequence")
        },
    }


def _tensor_analysis(
    tensor: LogicalTensor, topology: HybridParallelTopology
) -> dict[str, object]:
    shards = plan_tensor_shards(tensor, topology)
    reconstructed = reconstruct_tensor(shards)
    return {
        "shards": [
            {
                "tensor_rank": shard.tensor_rank,
                "tensor_parallel_size": shard.tensor_parallel_size,
                "original_shape": list(shard.original_shape),
                "offset": shard.offset,
                "padded_length": shard.padded_length,
                "padding_elements": shard.padding_elements,
                "values": list(shard.values),
            }
            for shard in shards
        ],
        "reconstructed": {
            "shape": list(reconstructed.shape),
            "values": list(reconstructed.values),
        },
    }


def _sequence_analysis(
    token_ids: tuple[int, ...], topology: HybridParallelTopology
) -> dict[str, object]:
    shards = plan_sequence_shards(token_ids, topology)
    reconstructed = reconstruct_sequence_shards(shards, topology)
    return {
        "shards": [
            {
                "sequence_rank": shard.sequence_rank,
                "sequence_size": shard.sequence_size,
                "original_length": shard.original_length,
                "padded_length": shard.padded_length,
                "start": shard.start,
                "token_ids": list(shard.token_ids),
                "padding_count": shard.padding_count,
            }
            for shard in shards
        ],
        "reconstructed_token_ids": list(reconstructed),
    }


def _build_report(
    *,
    topology: HybridParallelTopology,
    tensor: LogicalTensor,
    token_ids: tuple[int, ...],
    layer_count: int,
    schedule_policy: str,
    microbatch_count: int,
    communication_assumptions: CommunicationAssumptions,
    probe_observation: object,
) -> dict[str, object]:
    stage_plan = partition_transformer_layers(topology, layer_count=layer_count)
    if schedule_policy == "gpipe":
        schedule = generate_gpipe_schedule(
            stage_plan, microbatch_count=microbatch_count
        )
    elif schedule_policy == "1f1b":
        schedule = generate_1f1b_schedule(stage_plan, microbatch_count=microbatch_count)
    else:
        raise ValueError("schedule_policy must be gpipe or 1f1b")
    tensor_shards = plan_tensor_shards(tensor, topology)
    sequence_shards = plan_sequence_shards(token_ids, topology)
    admitted_probe = _validate_probe_observation(
        probe_observation, topology=topology, layer_count=layer_count
    )
    partition_mapping = {
        "layer_count": stage_plan.layer_count,
        "stages": [
            {
                "stage_index": stage.stage_index,
                "layer_indices": list(stage.layer_indices),
            }
            for stage in stage_plan.stages
        ],
        "per_stage_counts": list(stage_plan.per_stage_counts),
        "maximum_imbalance": stage_plan.maximum_imbalance,
    }
    return {
        "schema": REPORT_SCHEMA,
        "kind": REPORT_KIND,
        "inputs": {
            "topology": topology.to_mapping(),
            "tensor": {"shape": list(tensor.shape), "values": list(tensor.values)},
            "sequence": {"token_ids": list(token_ids)},
            "pipeline": {
                "layer_count": layer_count,
                "schedule_policy": schedule_policy,
                "microbatch_count": microbatch_count,
            },
            "communication_assumptions": communication_assumptions.to_mapping(),
            "probe_observation": admitted_probe,
        },
        "analytical": {
            "topology": _topology_analysis(topology),
            "tensor_shards": _tensor_analysis(tensor, topology),
            "sequence_shards": _sequence_analysis(token_ids, topology),
            "pipeline_partition": partition_mapping,
            "schedule": schedule.to_mapping(),
            "pipeline_bubbles": measure_pipeline_bubbles(schedule).to_mapping(),
            "communication": account_parallel_communication(
                topology=topology,
                tensor_shards=tensor_shards,
                sequence_shards=sequence_shards,
                schedule=schedule,
                assumptions=communication_assumptions,
            ).to_mapping(),
        },
        "executed_observations": {"pipeline_handoff": admitted_probe},
        "claims": {
            "analytical_planning": True,
            "cpu_gloo_transfer_conformance": True,
            "gpu_or_nvidia": False,
            "nccl": False,
            "performance": False,
            "production_readiness": False,
        },
        "evidence_boundary": EVIDENCE_BOUNDARY,
    }


def build_hybrid_parallel_report(
    *,
    topology: HybridParallelTopology,
    tensor: LogicalTensor,
    token_ids: tuple[int, ...],
    layer_count: int,
    schedule_policy: str,
    microbatch_count: int,
    communication_assumptions: CommunicationAssumptions,
    probe_result: PipelineHandoffProbeResult,
) -> dict[str, object]:
    """Build one unsealed report from admitted analytical and executed inputs."""

    if not isinstance(topology, HybridParallelTopology):
        raise TypeError("topology must be a HybridParallelTopology")
    if not isinstance(tensor, LogicalTensor):
        raise TypeError("tensor must be a LogicalTensor")
    if type(token_ids) is not tuple or any(
        type(token) is not int for token in token_ids
    ):
        raise TypeError("token_ids must be a tuple of integers")
    if not isinstance(communication_assumptions, CommunicationAssumptions):
        raise TypeError("communication_assumptions must be CommunicationAssumptions")
    if not isinstance(probe_result, PipelineHandoffProbeResult):
        raise TypeError("probe_result must be a verified PipelineHandoffProbeResult")
    try:
        return json.loads(
            _canonical_json_bytes(
                _build_report(
                    topology=topology,
                    tensor=tensor,
                    token_ids=token_ids,
                    layer_count=layer_count,
                    schedule_policy=schedule_policy,
                    microbatch_count=microbatch_count,
                    communication_assumptions=communication_assumptions,
                    probe_observation=probe_result.to_mapping(),
                )
            )
        )
    except HybridParallelReportError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report inputs cannot be admitted"
        ) from error


def _rebuild_from_inputs(inputs: object) -> dict[str, object]:
    mapping = _exact_mapping(inputs, _INPUT_FIELDS, name="report inputs")
    topology_payload = mapping["topology"]
    tensor_payload = mapping["tensor"]
    sequence_payload = mapping["sequence"]
    pipeline_payload = mapping["pipeline"]
    assumptions_payload = mapping["communication_assumptions"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            topology_payload,
            tensor_payload,
            sequence_payload,
            pipeline_payload,
            assumptions_payload,
        )
    ):
        raise ValueError("report input sections must be objects")
    _exact_mapping(tensor_payload, {"shape", "values"}, name="tensor input")
    _exact_mapping(sequence_payload, {"token_ids"}, name="sequence input")
    _exact_mapping(
        pipeline_payload,
        {"layer_count", "schedule_policy", "microbatch_count"},
        name="pipeline input",
    )
    topology = HybridParallelTopology.from_mapping(topology_payload)
    shape = tensor_payload["shape"]
    values = tensor_payload["values"]
    token_ids = sequence_payload["token_ids"]
    if not isinstance(shape, list) or not isinstance(values, list):
        raise ValueError("tensor inputs must use canonical list representations")
    if not isinstance(token_ids, list) or any(
        type(token) is not int for token in token_ids
    ):
        raise ValueError("sequence inputs must use an integer list")
    tensor = LogicalTensor(tuple(shape), tuple(values))
    assumptions = CommunicationAssumptions(**assumptions_payload)
    return _build_report(
        topology=topology,
        tensor=tensor,
        token_ids=tuple(token_ids),
        layer_count=pipeline_payload["layer_count"],
        schedule_policy=pipeline_payload["schedule_policy"],
        microbatch_count=pipeline_payload["microbatch_count"],
        communication_assumptions=assumptions,
        probe_observation=mapping["probe_observation"],
    )


def verify_hybrid_parallel_report(
    report: dict[str, object],
) -> dict[str, object]:
    """Verify the seal, evidence boundaries, and every replayable field."""

    if not isinstance(report, dict):
        raise TypeError("hybrid parallel report must be a dictionary")
    if set(report) != _TOP_LEVEL_FIELDS:
        raise HybridParallelReportError(
            "hybrid parallel report fields do not match the canonical schema"
        )
    if report.get("schema") != REPORT_SCHEMA:
        raise HybridParallelReportError("unsupported hybrid parallel report schema")
    if report.get("kind") != REPORT_KIND:
        raise HybridParallelReportError("unexpected hybrid parallel report kind")
    try:
        expected_integrity = seal_hybrid_parallel_report(report)["integrity"]
    except HybridParallelReportError:
        raise
    if report.get("integrity") != expected_integrity:
        raise HybridParallelReportError("hybrid parallel report integrity mismatch")
    inputs = report.get("inputs")
    observations = report.get("executed_observations")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(observations, Mapping)
        or set(observations) != {"pipeline_handoff"}
        or observations["pipeline_handoff"] != inputs.get("probe_observation")
    ):
        raise HybridParallelReportError(
            "executed probe observation drifted from primary evidence"
        )
    try:
        expected = seal_hybrid_parallel_report(_rebuild_from_inputs(inputs))
    except HybridParallelReportError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report cannot be rebuilt from primary inputs"
        ) from error
    if report != expected:
        raise HybridParallelReportError(
            "hybrid parallel report derived fields drifted from primary inputs"
        )
    return report
