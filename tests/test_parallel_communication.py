"""Hybrid communication accounting keeps logical and wire bytes distinct."""

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.parallel_communication import (
    CommunicationAssumptions,
    account_parallel_communication,
)
from ai_journey.pipeline_partition import partition_transformer_layers
from ai_journey.pipeline_schedule import generate_1f1b_schedule, generate_gpipe_schedule
from ai_journey.sequence_sharding import plan_sequence_shards
from ai_journey.tensor_sharding import LogicalTensor, plan_tensor_shards


def _admitted_inputs():
    topology = HybridParallelTopology(8, 1, 2, 2, 2)
    tensor_shards = plan_tensor_shards(
        LogicalTensor(shape=(5,), values=(0, 1, 2, 3, 4)), topology
    )
    sequence_shards = plan_sequence_shards((11, 12, 13, 14, 15), topology)
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)
    return topology, tensor_shards, sequence_shards, schedule


def test_accounting_separates_tensor_sequence_and_pipeline_volume():
    topology, tensor_shards, sequence_shards, schedule = _admitted_inputs()

    accounting = account_parallel_communication(
        topology=topology,
        tensor_shards=tensor_shards,
        sequence_shards=sequence_shards,
        schedule=schedule,
        assumptions=CommunicationAssumptions(
            element_size_bytes=4,
            pipeline_payload_bytes=32,
        ),
    )

    assert accounting.to_mapping() == {
        "assumptions": {
            "element_size_bytes": 4,
            "pipeline_payload_bytes": 32,
        },
        "wire_model": {
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
        },
        "tensor_collective": {
            "dimension": "tensor",
            "operation": "ring_all_gather",
            "lane_count": 4,
            "operations_per_lane": 1,
            "operation_count": 4,
            "payload_bytes_per_operation": 24,
            "logical_bytes_per_lane": 20,
            "logical_bytes": 80,
            "wire_bytes_per_lane": 24,
            "wire_bytes": 96,
        },
        "sequence_transfer": {
            "dimension": "sequence",
            "operation": "all_to_all",
            "lane_count": 4,
            "operations_per_lane": 1,
            "operation_count": 4,
            "payload_bytes_per_operation": 24,
            "logical_bytes_per_lane": 20,
            "logical_bytes": 80,
            "wire_bytes_per_lane": 16,
            "wire_bytes": 64,
        },
        "pipeline_handoff": {
            "dimension": "pipeline",
            "operation": "point_to_point",
            "lane_count": 4,
            "operations_per_lane": 4,
            "operation_count": 16,
            "payload_bytes_per_operation": 32,
            "logical_bytes_per_lane": 128,
            "logical_bytes": 512,
            "wire_bytes_per_lane": 128,
            "wire_bytes": 512,
        },
        "evidence_boundary": (
            "Analytical communication accounting from admitted plans and explicit "
            "wire assumptions; it does not execute collectives, measure bandwidth, "
            "or establish accelerator, NCCL, multi-node, performance, or production evidence."
        ),
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("element_size_bytes", -1, "element_size_bytes must be positive"),
        ("pipeline_payload_bytes", -1, "pipeline_payload_bytes must be positive"),
    ],
)
def test_assumptions_reject_negative_sizes(field, value, message):
    values = {"element_size_bytes": 4, "pipeline_payload_bytes": 32, field: value}

    with pytest.raises(ValueError, match=message):
        CommunicationAssumptions(**values)


def test_accounting_requires_public_admitted_input_types():
    topology, tensor_shards, sequence_shards, _ = _admitted_inputs()
    assumptions = CommunicationAssumptions(4, 32)

    with pytest.raises(TypeError, match="schedule must be a PipelineSchedule"):
        account_parallel_communication(
            topology=topology,
            tensor_shards=tensor_shards,
            sequence_shards=sequence_shards,
            schedule=object(),
            assumptions=assumptions,
        )


def test_accounting_rejects_plans_incompatible_with_the_admitted_topology():
    topology, tensor_shards, sequence_shards, _ = _admitted_inputs()
    one_stage_topology = HybridParallelTopology(4, 1, 2, 1, 2)
    one_stage_plan = partition_transformer_layers(one_stage_topology, layer_count=2)
    incompatible_schedule = generate_gpipe_schedule(one_stage_plan, microbatch_count=2)

    with pytest.raises(ValueError, match="pipeline schedule is incompatible"):
        account_parallel_communication(
            topology=topology,
            tensor_shards=tensor_shards,
            sequence_shards=sequence_shards,
            schedule=incompatible_schedule,
            assumptions=CommunicationAssumptions(4, 32),
        )


def test_byte_overflow_is_rejected_instead_of_emitting_unbounded_evidence():
    topology, tensor_shards, sequence_shards, schedule = _admitted_inputs()

    with pytest.raises(ValueError, match="calculation exceeds"):
        account_parallel_communication(
            topology=topology,
            tensor_shards=tensor_shards,
            sequence_shards=sequence_shards,
            schedule=schedule,
            assumptions=CommunicationAssumptions((1 << 63) - 1, 1),
        )


def test_single_rank_dimensions_have_no_assumed_wire_traffic():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)
    tensor_shards = plan_tensor_shards(LogicalTensor((2,), (1, 2)), topology)
    sequence_shards = plan_sequence_shards((3, 4), topology)
    stage_plan = partition_transformer_layers(topology, layer_count=1)
    schedule = generate_gpipe_schedule(stage_plan, microbatch_count=2)

    accounting = account_parallel_communication(
        topology=topology,
        tensor_shards=tensor_shards,
        sequence_shards=sequence_shards,
        schedule=schedule,
        assumptions=CommunicationAssumptions(2, 16),
    )

    assert accounting.tensor_collective.wire_bytes == 0
    assert accounting.sequence_transfer.wire_bytes == 0
    assert accounting.tensor_collective.lane_count == 1
    assert accounting.sequence_transfer.lane_count == 1
    assert accounting.pipeline_handoff.operation_count == 0
    assert accounting.pipeline_handoff.wire_bytes == 0


def test_pipeline_accounting_uses_dependencies_not_schedule_policy_slots():
    topology, tensor_shards, sequence_shards, gpipe = _admitted_inputs()
    stage_plan = partition_transformer_layers(topology, layer_count=4)
    one_f_one_b = generate_1f1b_schedule(stage_plan, microbatch_count=2)
    assumptions = CommunicationAssumptions(4, 32)

    gpipe_result = account_parallel_communication(
        topology=topology,
        tensor_shards=tensor_shards,
        sequence_shards=sequence_shards,
        schedule=gpipe,
        assumptions=assumptions,
    )
    one_f_one_b_result = account_parallel_communication(
        topology=topology,
        tensor_shards=tensor_shards,
        sequence_shards=sequence_shards,
        schedule=one_f_one_b,
        assumptions=assumptions,
    )

    assert gpipe_result.pipeline_handoff == one_f_one_b_result.pipeline_handoff
