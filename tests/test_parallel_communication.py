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
            "tensor_collective": "ring_all_gather_equal_ceiling_chunks",
            "sequence_transfer": "direct_all_to_all_equal_ceiling_chunks",
            "pipeline_handoff": "point_to_point_cross_stage_dependencies",
            "wire_scope": "aggregate_across_participating_ranks",
        },
        "tensor_collective": {
            "dimension": "tensor",
            "operation": "ring_all_gather",
            "operation_count": 1,
            "payload_bytes_per_operation": 24,
            "logical_bytes": 20,
            "wire_bytes": 24,
        },
        "sequence_transfer": {
            "dimension": "sequence",
            "operation": "all_to_all",
            "operation_count": 1,
            "payload_bytes_per_operation": 24,
            "logical_bytes": 20,
            "wire_bytes": 16,
        },
        "pipeline_handoff": {
            "dimension": "pipeline",
            "operation": "point_to_point",
            "operation_count": 4,
            "payload_bytes_per_operation": 32,
            "logical_bytes": 128,
            "wire_bytes": 128,
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
        ("tensor_collective", "tree_all_gather", "unsupported tensor collective"),
        ("sequence_transfer", "ring_exchange", "unsupported sequence transfer"),
        ("pipeline_handoff", "broadcast", "unsupported pipeline handoff"),
    ],
)
def test_assumptions_reject_negative_sizes_and_unsupported_wire_models(
    field, value, message
):
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
