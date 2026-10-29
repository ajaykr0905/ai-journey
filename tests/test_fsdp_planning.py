import pytest

from ai_journey.fsdp_planning import (
    FSDPPlanningError,
    decoder_parameter_shards,
    partition_parameter_groups,
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


def test_parameter_partition_rejects_empty_and_ambiguous_inputs():
    with pytest.raises(FSDPPlanningError, match="nonempty"):
        partition_parameter_groups({}, world_size=1)
    with pytest.raises(FSDPPlanningError, match="contain bytes"):
        partition_parameter_groups({"empty": 0}, world_size=1)
    with pytest.raises(TypeError, match="world_size"):
        partition_parameter_groups({"model": 1}, world_size=True)
    with pytest.raises(TypeError, match="sizes"):
        partition_parameter_groups({"model": True}, world_size=1)
