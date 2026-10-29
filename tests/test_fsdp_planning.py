import pytest

from ai_journey.fsdp_planning import (
    FSDPPlanningError,
    all_gather_transient_peaks,
    decoder_parameter_shards,
    partition_parameter_groups,
    sharded_training_state,
)
from ai_journey.resource_accounting import DecoderSpec


def test_flat_parameter_partition_is_balanced_lossless_and_ordered():
    shards = partition_parameter_groups(
        {"embedding": 5, "blocks": 8, "empty": 0, "head": 4}, world_size=3
    )

    assert [shard.payload_bytes for shard in shards] == [6, 6, 5]
    assert [shard.rank for shard in shards] == [0, 1, 2]
    assert sum(shard.payload_bytes for shard in shards) == 17
    assert [
        (part.parameter_group, part.offset_bytes, part.length_bytes)
        for shard in shards
        for part in shard.slices
    ] == [
        ("embedding", 0, 5),
        ("blocks", 0, 1),
        ("blocks", 1, 6),
        ("blocks", 7, 1),
        ("head", 0, 4),
    ]


def test_decoder_parameter_partition_uses_declared_dtype_width():
    model = DecoderSpec(32, 2, 16, 4, 2, 32)
    shards = decoder_parameter_shards(model, dtype="bfloat16", world_size=4)
    assert sum(shard.payload_bytes for shard in shards) == model.parameter_count * 2
    assert (
        max(shard.payload_bytes for shard in shards)
        - min(shard.payload_bytes for shard in shards)
        <= 1
    )


def test_all_gather_peak_counts_only_missing_group_bytes():
    groups = {"embedding": 5, "blocks": 8, "head": 4}
    shards = partition_parameter_groups(groups, world_size=3)

    peaks = all_gather_transient_peaks(groups, shards)

    assert [peak.rank for peak in peaks] == [0, 1, 2]
    assert [peak.parameter_group for peak in peaks] == ["blocks", "embedding", "blocks"]
    assert [peak.local_bytes for peak in peaks] == [1, 0, 1]
    assert [peak.materialized_bytes for peak in peaks] == [8, 5, 8]
    assert [peak.transient_bytes for peak in peaks] == [7, 5, 7]


def test_all_gather_peak_rejects_noncanonical_shards():
    groups = {"model": 9}
    shards = partition_parameter_groups(groups, world_size=2)
    with pytest.raises(FSDPPlanningError, match="canonical"):
        all_gather_transient_peaks(groups, tuple(reversed(shards)))
    with pytest.raises(FSDPPlanningError, match="nonempty"):
        all_gather_transient_peaks(groups, ())


def test_parameter_partition_rejects_empty_and_ambiguous_inputs():
    with pytest.raises(FSDPPlanningError, match="nonempty"):
        partition_parameter_groups({}, world_size=1)
    with pytest.raises(FSDPPlanningError, match="contain bytes"):
        partition_parameter_groups({"empty": 0}, world_size=1)
    with pytest.raises(TypeError, match="world_size"):
        partition_parameter_groups({"model": 1}, world_size=True)
    with pytest.raises(TypeError, match="sizes"):
        partition_parameter_groups({"model": True}, world_size=1)


def test_sharded_training_state_accounts_for_each_declared_copy():
    model = DecoderSpec(32, 2, 16, 4, 2, 32)
    states = sharded_training_state(
        model,
        parameter_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
        world_size=3,
    )

    assert sum(state.parameters_bytes for state in states) == model.parameter_count * 2
    assert sum(state.gradients_bytes for state in states) == model.parameter_count * 4
    assert sum(state.optimizer_bytes for state in states) == model.parameter_count * 8
    assert sum(state.master_parameters_bytes for state in states) == (
        model.parameter_count * 4
    )
    assert sum(state.total_bytes for state in states) == model.parameter_count * 18


def test_sharded_training_state_can_omit_master_parameters_and_optimizer_slots():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    states = sharded_training_state(
        model,
        parameter_dtype="float32",
        gradient_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=2,
    )
    assert all(state.optimizer_bytes == 0 for state in states)
    assert all(state.master_parameters_bytes == 0 for state in states)


def test_sharded_training_state_rejects_ambiguous_optimizer_slots():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    with pytest.raises(TypeError, match="optimizer_slots"):
        sharded_training_state(
            model,
            parameter_dtype="float32",
            gradient_dtype="float32",
            optimizer_dtype="float32",
            optimizer_slots=True,
            world_size=1,
        )
