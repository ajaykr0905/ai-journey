"""Deterministic analytical sections for admitted hybrid-parallel inputs."""

from __future__ import annotations

from .hybrid_topology import HybridParallelTopology
from .parallel_communication import (
    CommunicationAssumptions,
    account_parallel_communication,
)
from .pipeline_metrics import measure_pipeline_bubbles
from .pipeline_partition import partition_transformer_layers
from .pipeline_schedule import generate_1f1b_schedule, generate_gpipe_schedule
from .sequence_sharding import plan_sequence_shards, reconstruct_sequence_shards
from .tensor_sharding import LogicalTensor, plan_tensor_shards, reconstruct_tensor


def build_analytical_results(
    *,
    topology: HybridParallelTopology,
    tensor: LogicalTensor,
    token_ids: tuple[int, ...],
    layer_count: int,
    schedule_policy: str,
    microbatch_count: int,
    communication_assumptions: CommunicationAssumptions,
) -> dict[str, object]:
    """Build all replayable analytical fields after bounded admission."""

    stage_plan = partition_transformer_layers(topology, layer_count=layer_count)
    schedule = (
        generate_gpipe_schedule(stage_plan, microbatch_count=microbatch_count)
        if schedule_policy == "gpipe"
        else generate_1f1b_schedule(stage_plan, microbatch_count=microbatch_count)
    )
    tensor_shards = plan_tensor_shards(tensor, topology)
    sequence_shards = plan_sequence_shards(token_ids, topology)
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
    reconstructed_tensor = reconstruct_tensor(tensor_shards)
    reconstructed_sequence = reconstruct_sequence_shards(sequence_shards, topology)
    return {
        "topology": {
            "coordinates": coordinates,
            "peer_groups": {
                dimension: [list(group) for group in topology.peer_groups(dimension)]
                for dimension in ("data", "tensor", "pipeline", "sequence")
            },
        },
        "tensor_shards": {
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
                for shard in tensor_shards
            ],
            "reconstructed": {
                "shape": list(reconstructed_tensor.shape),
                "values": list(reconstructed_tensor.values),
            },
        },
        "sequence_shards": {
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
                for shard in sequence_shards
            ],
            "reconstructed_token_ids": list(reconstructed_sequence),
        },
        "pipeline_partition": {
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
        },
        "schedule": schedule.to_mapping(),
        "pipeline_bubbles": measure_pipeline_bubbles(schedule).to_mapping(),
        "communication": account_parallel_communication(
            topology=topology,
            tensor_shards=tensor_shards,
            sequence_shards=sequence_shards,
            schedule=schedule,
            assumptions=communication_assumptions,
        ).to_mapping(),
    }
