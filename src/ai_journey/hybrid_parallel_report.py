"""Assemble bounded hybrid-parallel analytical and recorded execution evidence."""

from __future__ import annotations

import json

from .hybrid_parallel_analysis import build_analytical_results
from .hybrid_parallel_schema import (
    EVIDENCE_BOUNDARY,
    REPORT_KIND,
    REPORT_SCHEMA,
    HybridParallelReportError,
    admit_generation_bounds,
)
from .hybrid_parallel_seal import (
    canonical_hybrid_parallel_report_bytes,
    seal_hybrid_parallel_report,
)
from .hybrid_topology import HybridParallelTopology
from .parallel_communication import CommunicationAssumptions
from .pipeline_handoff_probe import PipelineHandoffProbeResult
from .tensor_sharding import LogicalTensor


def assemble_hybrid_parallel_report(
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
    """Assemble one report after bounds and probe schema are admitted."""

    admit_generation_bounds(
        topology=topology,
        tensor=tensor,
        token_ids=token_ids,
        layer_count=layer_count,
        microbatch_count=microbatch_count,
    )
    if schedule_policy not in ("gpipe", "1f1b"):
        raise ValueError("schedule_policy must be gpipe or 1f1b")
    admitted_probe = PipelineHandoffProbeResult.admit_mapping(
        probe_observation, topology=topology, layer_count=layer_count
    )
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
        "analytical": build_analytical_results(
            topology=topology,
            tensor=tensor,
            token_ids=token_ids,
            layer_count=layer_count,
            schedule_policy=schedule_policy,
            microbatch_count=microbatch_count,
            communication_assumptions=communication_assumptions,
        ),
        "executed_observations": {"pipeline_handoff": admitted_probe},
        "claims": {
            "analytical_planning": True,
            "cpu_gloo_execution_recorded": True,
            "independent_execution_authenticity": False,
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
    """Build one unsealed report from analytical inputs and a recorded probe."""

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
        raise TypeError("probe_result must be a PipelineHandoffProbeResult")
    try:
        return json.loads(
            canonical_hybrid_parallel_report_bytes(
                assemble_hybrid_parallel_report(
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
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HybridParallelReportError(
            "hybrid parallel report inputs cannot be admitted"
        ) from error


def verify_hybrid_parallel_report(report: dict[str, object]) -> dict[str, object]:
    """Compatibility export for the focused offline verifier."""

    from .hybrid_parallel_report_verification import verify_hybrid_parallel_report

    return verify_hybrid_parallel_report(report)


__all__ = [
    "HybridParallelReportError",
    "build_hybrid_parallel_report",
    "canonical_hybrid_parallel_report_bytes",
    "seal_hybrid_parallel_report",
    "verify_hybrid_parallel_report",
]
