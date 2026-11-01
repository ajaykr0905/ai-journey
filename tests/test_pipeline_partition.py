"""Transformer layers are assigned to contiguous pipeline stages."""

import pytest

from ai_journey.hybrid_topology import HybridParallelTopology
from ai_journey.pipeline_partition import partition_transformer_layers


def test_uneven_layers_are_owned_once_in_order_with_imbalance_evidence():
    topology = HybridParallelTopology(6, 2, 1, 3, 1)

    plan = partition_transformer_layers(topology, layer_count=10)

    assert tuple(stage.stage_index for stage in plan.stages) == (0, 1, 2)
    assert tuple(stage.layer_indices for stage in plan.stages) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )
    assert plan.per_stage_counts == (4, 3, 3)
    assert plan.maximum_imbalance == 1
    assert tuple(
        layer for stage in plan.stages for layer in stage.layer_indices
    ) == tuple(range(10))


def test_pipeline_partition_rejects_plans_with_empty_stages():
    topology = HybridParallelTopology(4, 1, 1, 4, 1)

    with pytest.raises(ValueError, match="cannot leave pipeline stages empty"):
        partition_transformer_layers(topology, layer_count=3)


def test_balanced_partition_is_deterministic_and_reports_zero_imbalance():
    topology = HybridParallelTopology(2, 1, 1, 2, 1)

    first = partition_transformer_layers(topology, layer_count=6)
    second = partition_transformer_layers(topology, layer_count=6)

    assert first == second
    assert first.per_stage_counts == (3, 3)
    assert first.maximum_imbalance == 0


def test_pipeline_partition_requires_admitted_primary_inputs():
    topology = HybridParallelTopology(1, 1, 1, 1, 1)

    with pytest.raises(TypeError, match="topology must be a HybridParallelTopology"):
        partition_transformer_layers(object(), layer_count=1)
    with pytest.raises(TypeError, match="layer_count must be an integer"):
        partition_transformer_layers(topology, layer_count=True)
    with pytest.raises(ValueError, match="layer_count must be positive"):
        partition_transformer_layers(topology, layer_count=0)
