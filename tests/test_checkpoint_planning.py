import pytest

from ai_journey.checkpoint_planning import (
    CheckpointPlanningError,
    checkpoint_image,
)
from ai_journey.resource_accounting import DecoderSpec


def test_sharded_checkpoint_preserves_every_declared_state_byte():
    model = DecoderSpec(32, 2, 16, 4, 2, 32)
    image = checkpoint_image(
        model,
        parameter_dtype="bfloat16",
        optimizer_dtype="float32",
        optimizer_slots=2,
        master_parameter_dtype="float32",
        world_size=3,
        sharded=True,
        rank_local_bytes=17,
        metadata_bytes=29,
    )
    expected_global = model.parameter_count * (2 + 8 + 4)
    assert image.total_bytes == expected_global + 3 * 17 + 29
    payloads = [file.payload_bytes for file in image.rank_files]
    assert max(payloads) - min(payloads) <= 3


def test_full_checkpoint_writes_global_state_only_on_coordinator():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    image = checkpoint_image(
        model,
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=2,
        sharded=False,
        rank_local_bytes=11,
    )
    assert image.rank_files[0].payload_bytes == model.parameter_count * 4 + 11
    assert image.rank_files[1].payload_bytes == 11
    assert image.total_bytes == model.parameter_count * 4 + 22


def test_checkpoint_image_rejects_ambiguous_or_negative_controls():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    with pytest.raises(TypeError, match="world_size"):
        checkpoint_image(
            model,
            parameter_dtype="float32",
            optimizer_dtype="float32",
            optimizer_slots=2,
            world_size=True,
            sharded=True,
        )
    with pytest.raises(CheckpointPlanningError, match="nonnegative"):
        checkpoint_image(
            model,
            parameter_dtype="float32",
            optimizer_dtype="float32",
            optimizer_slots=-1,
            world_size=1,
            sharded=True,
        )
