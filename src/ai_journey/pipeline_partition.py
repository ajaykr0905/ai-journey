"""Deterministic transformer-layer ownership across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from .hybrid_topology import HybridParallelTopology


@dataclass(frozen=True)
class PipelineStageAssignment:
    """The ordered transformer layers owned by one pipeline stage."""

    stage_index: int
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class PipelinePartitionPlan:
    """Contiguous stage ownership and its observable balance evidence."""

    layer_count: int
    stages: tuple[PipelineStageAssignment, ...]

    @property
    def per_stage_counts(self) -> tuple[int, ...]:
        """Return the number of transformer layers assigned to each stage."""

        return tuple(len(stage.layer_indices) for stage in self.stages)

    @property
    def maximum_imbalance(self) -> int:
        """Return the largest difference between stage layer counts."""

        counts = self.per_stage_counts
        return max(counts) - min(counts)


def partition_transformer_layers(
    topology: HybridParallelTopology, *, layer_count: int
) -> PipelinePartitionPlan:
    """Assign ordered transformer layers contiguously across pipeline stages."""

    if not isinstance(topology, HybridParallelTopology):
        raise TypeError("topology must be a HybridParallelTopology")
    if type(layer_count) is not int:
        raise TypeError("layer_count must be an integer")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    stage_count = topology.pipeline_parallel_size
    if layer_count < stage_count:
        raise ValueError("layer_count cannot leave pipeline stages empty")
    base_count, remainder = divmod(layer_count, stage_count)
    stages = []
    start = 0
    for stage_index in range(stage_count):
        count = base_count + (stage_index < remainder)
        stop = start + count
        stages.append(PipelineStageAssignment(stage_index, tuple(range(start, stop))))
        start = stop
    return PipelinePartitionPlan(layer_count=layer_count, stages=tuple(stages))
