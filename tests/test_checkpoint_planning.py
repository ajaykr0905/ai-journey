import pytest

from ai_journey.checkpoint_planning import (
    CheckpointImage,
    CheckpointPlanningError,
    checkpoint_image,
    checkpoint_retention,
    checkpoint_write_estimate,
    recovery_estimate,
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


def test_checkpoint_write_estimate_respects_shared_bandwidth_floor():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    image = checkpoint_image(
        model,
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=2,
        sharded=True,
        metadata_bytes=100,
    )
    estimate = checkpoint_write_estimate(
        image,
        rank_bandwidth_bytes_per_second=200,
        shared_bandwidth_bytes_per_second=100,
        metadata_bandwidth_bytes_per_second=50,
    )
    assert estimate.shared_bandwidth_floor_seconds == model.parameter_count * 4 / 100
    assert estimate.metadata_write_seconds == 2
    assert estimate.total_seconds == estimate.shared_bandwidth_floor_seconds + 2


def test_checkpoint_write_estimate_rejects_nonfinite_rates():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    image = checkpoint_image(
        model,
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=1,
        sharded=False,
    )
    with pytest.raises(CheckpointPlanningError, match="finite"):
        checkpoint_write_estimate(
            image,
            rank_bandwidth_bytes_per_second=float("inf"),
            shared_bandwidth_bytes_per_second=100,
        )


def test_checkpoint_retention_never_counts_partial_images():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    image = checkpoint_image(
        model,
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=1,
        sharded=False,
    )
    budget = image.total_bytes * 3 + image.total_bytes // 2
    retention = checkpoint_retention(image, storage_budget_bytes=budget)
    assert retention.retained_images == 3
    assert retention.consumed_bytes == image.total_bytes * 3
    assert retention.remaining_bytes == image.total_bytes // 2
    assert retention.covers_interval(100, 300)
    assert not retention.covers_interval(100, 301)


def test_checkpoint_retention_rejects_empty_images_and_negative_budgets():
    empty = checkpoint_image(
        DecoderSpec(32, 1, 8, 2, 1, 16),
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=1,
        sharded=False,
    )
    with pytest.raises(CheckpointPlanningError, match="nonnegative"):
        checkpoint_retention(empty, storage_budget_bytes=-1)
    with pytest.raises(CheckpointPlanningError, match="contain bytes"):
        checkpoint_retention(
            CheckpointImage(sharded=False, rank_files=(), metadata_bytes=0),
            storage_budget_bytes=100,
        )


def test_recovery_estimate_combines_load_restart_and_replay():
    model = DecoderSpec(32, 1, 8, 2, 1, 16)
    image = checkpoint_image(
        model,
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=2,
        sharded=True,
    )
    estimate = recovery_estimate(
        image,
        rank_read_bandwidth_bytes_per_second=200,
        shared_read_bandwidth_bytes_per_second=100,
        lost_optimizer_steps=3,
        seconds_per_optimizer_step=2.5,
        restart_seconds=4,
    )
    assert estimate.checkpoint_load_seconds == image.total_bytes / 100
    assert estimate.replay_seconds == 7.5
    assert estimate.total_seconds == image.total_bytes / 100 + 11.5
    assert estimate.meets_rto(estimate.total_seconds)
    assert not estimate.meets_rto(estimate.total_seconds - 0.1)


def test_recovery_estimate_rejects_negative_lost_work():
    image = checkpoint_image(
        DecoderSpec(32, 1, 8, 2, 1, 16),
        parameter_dtype="float32",
        optimizer_dtype="float32",
        optimizer_slots=0,
        world_size=1,
        sharded=False,
    )
    with pytest.raises(CheckpointPlanningError, match="lost_optimizer_steps"):
        recovery_estimate(
            image,
            rank_read_bandwidth_bytes_per_second=100,
            shared_read_bandwidth_bytes_per_second=100,
            lost_optimizer_steps=-1,
            seconds_per_optimizer_step=1,
        )
