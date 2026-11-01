"""Hybrid parallel reports replay every analytical field offline."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ai_journey.activation_handoff import ActivationHandoffContract
from ai_journey.hybrid_parallel_report import (
    HybridParallelReportError,
    build_hybrid_parallel_report,
    seal_hybrid_parallel_report,
    verify_hybrid_parallel_report,
)
from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.parallel_communication import CommunicationAssumptions
from ai_journey.pipeline_handoff_probe import PipelineHandoffProbeResult
from ai_journey.pipeline_partition import partition_transformer_layers
from ai_journey.tensor_sharding import LogicalTensor


def report_inputs() -> dict[str, object]:
    topology = HybridParallelTopology(8, 1, 2, 2, 2)
    stage_plan = partition_transformer_layers(topology, layer_count=5)
    payload = bytes(range(16))
    contract = ActivationHandoffContract.admit(
        topology=topology,
        partition=stage_plan,
        source_rank=0,
        destination_rank=2,
        tensor_shape=(4,),
        dtype="float32",
        payload=payload,
    )
    probe = PipelineHandoffProbeResult(
        contract,
        (
            {"probe_rank": 0, "role": "source", "outcome": "success"},
            {"probe_rank": 1, "role": "destination", "outcome": "success"},
        ),
    )
    return {
        "topology": topology,
        "tensor": LogicalTensor((2, 3), tuple(range(6))),
        "token_ids": (11, 12, 13, 14, 15),
        "layer_count": 5,
        "schedule_policy": "1f1b",
        "microbatch_count": 2,
        "communication_assumptions": CommunicationAssumptions(4, 16),
        "probe_result": probe,
    }


def sealed_report() -> dict[str, object]:
    return seal_hybrid_parallel_report(build_hybrid_parallel_report(**report_inputs()))


def test_build_report_separates_replayable_analysis_from_executed_probe():
    report = sealed_report()

    assert verify_hybrid_parallel_report(report) is report
    assert report["schema"] == "ai-journey.hybrid-parallel-report.v1"
    assert report["analytical"]["topology"]["coordinates"][2] == {
        "rank": 2,
        "data": 0,
        "tensor": 0,
        "pipeline": 1,
        "sequence": 0,
    }
    assert report["analytical"]["tensor_shards"]["reconstructed"] == {
        "shape": [2, 3],
        "values": [0, 1, 2, 3, 4, 5],
    }
    assert report["analytical"]["sequence_shards"]["reconstructed_token_ids"] == [
        11,
        12,
        13,
        14,
        15,
    ]
    assert report["analytical"]["pipeline_partition"]["per_stage_counts"] == [3, 2]
    assert report["analytical"]["schedule"]["policy"] == "1f1b"
    assert report["analytical"]["pipeline_bubbles"]["policy"] == "1f1b"
    assert (
        report["analytical"]["communication"]["pipeline_handoff"]["operation_count"]
        == 16
    )
    assert report["analytical"]["communication"]["pipeline_handoff"]["lane_count"] == 4
    assert report["analytical"]["communication"]["wire_model"] == {
        "aggregation_scope": "aggregate_all_topology_lanes",
        "tensor_collective": "ring_all_gather_equal_ceiling_shards",
        "tensor_wire_formula": (
            "lane_count * tensor_parallel_size * (tensor_parallel_size - 1) "
            "* padded_shard_elements * element_size_bytes"
        ),
        "sequence_transfer": "direct_all_to_all_equal_ceiling_segments",
        "sequence_wire_formula": (
            "lane_count * sequence_parallel_size * (sequence_parallel_size - 1) "
            "* ceil(shard_width / sequence_parallel_size) * element_size_bytes"
        ),
        "pipeline_handoff": "point_to_point_cross_stage_dependencies",
        "pipeline_wire_formula": (
            "lane_count * cross_stage_dependency_count * pipeline_payload_bytes"
        ),
    }
    assert report["executed_observations"]["pipeline_handoff"]["backend"] == "gloo"
    assert report["executed_observations"]["pipeline_handoff"]["device"] == "cpu"
    assert report["claims"] == {
        "analytical_planning": True,
        "cpu_gloo_execution_recorded": True,
        "independent_execution_authenticity": False,
        "gpu_or_nvidia": False,
        "nccl": False,
        "performance": False,
        "production_readiness": False,
    }
    boundary = report["evidence_boundary"]
    assert "CPU/Gloo" in boundary
    assert all(
        term in boundary for term in ("NVIDIA", "NCCL", "performance", "production")
    )
    assert "does not independently authenticate execution" in boundary


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("topology", lambda value: value["coordinates"][0].update({"pipeline": 1})),
        ("tensor_shards", lambda value: value["reconstructed"]["values"].append(99)),
        (
            "sequence_shards",
            lambda value: value["reconstructed_token_ids"].append(99),
        ),
        (
            "pipeline_partition",
            lambda value: value.update({"maximum_imbalance": 99}),
        ),
        ("schedule", lambda value: value.update({"policy": "gpipe"})),
        (
            "pipeline_bubbles",
            lambda value: value["utilization"].update({"value": 1.0}),
        ),
        (
            "communication",
            lambda value: value["pipeline_handoff"].update({"wire_bytes": 1}),
        ),
    ],
)
def test_resealed_analytical_drift_is_rejected(section, mutate):
    report = sealed_report()
    mutate(report["analytical"][section])
    report = seal_hybrid_parallel_report(report)

    with pytest.raises(HybridParallelReportError, match="derived fields drifted"):
        verify_hybrid_parallel_report(report)


def test_resealed_probe_claim_escalation_is_rejected_without_rerunning_gloo():
    report = sealed_report()
    observation = report["executed_observations"]["pipeline_handoff"]
    observation["backend"] = "nccl"
    observation["device"] = "cuda"
    observation["performance_measured"] = True
    report = seal_hybrid_parallel_report(report)

    with pytest.raises(HybridParallelReportError, match="probe observation"):
        verify_hybrid_parallel_report(report)


def test_report_verification_rejects_schema_integrity_and_primary_input_drift():
    report = sealed_report()
    wrong_schema = deepcopy(report)
    wrong_schema["schema"] = "ai-journey.hybrid-parallel-report.v2"
    wrong_schema = seal_hybrid_parallel_report(wrong_schema)
    with pytest.raises(HybridParallelReportError, match="schema"):
        verify_hybrid_parallel_report(wrong_schema)

    wrong_digest = deepcopy(report)
    wrong_digest["integrity"]["digest"] = "0" * 64
    with pytest.raises(HybridParallelReportError, match="integrity"):
        verify_hybrid_parallel_report(wrong_digest)

    invalid_input = deepcopy(report)
    invalid_input["inputs"]["topology"]["world_size"] = 7
    invalid_input = seal_hybrid_parallel_report(invalid_input)
    with pytest.raises(HybridParallelReportError, match="cannot be rebuilt"):
        verify_hybrid_parallel_report(invalid_input)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda inputs: inputs["topology"].update(
                {
                    "world_size": 1_000_000,
                    "data_parallel_size": 1_000_000,
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "sequence_parallel_size": 1,
                }
            ),
            "world_size exceeds",
        ),
        (
            lambda inputs: inputs["pipeline"].update({"layer_count": 1_000_000}),
            "layer_count exceeds",
        ),
        (
            lambda inputs: inputs["pipeline"].update({"microbatch_count": 1_000_000}),
            "microbatch_count exceeds",
        ),
    ],
)
def test_report_verification_applies_caps_before_analytical_materialization(
    mutate, message, monkeypatch
):
    report = sealed_report()
    mutate(report["inputs"])
    report = seal_hybrid_parallel_report(report)

    def unexpected_materialization(*args, **kwargs):
        raise AssertionError("untrusted oversized inputs must reject before expansion")

    monkeypatch.setattr(
        HybridParallelTopology, "coordinate_for_rank", unexpected_materialization
    )
    with pytest.raises(HybridParallelReportError, match=message):
        verify_hybrid_parallel_report(report)
