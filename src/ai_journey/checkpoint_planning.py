"""Deterministic checkpoint capacity and recovery planning primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .resource_accounting import DecoderSpec, bytes_per_element


class CheckpointPlanningError(ValueError):
    """Raised when a checkpoint planning assumption is invalid."""


@dataclass(frozen=True)
class RankCheckpointFile:
    """One rank's checkpoint payload decomposition."""

    rank: int
    components: tuple[tuple[str, int], ...]

    @property
    def payload_bytes(self) -> int:
        return sum(size for _, size in self.components)


@dataclass(frozen=True)
class CheckpointImage:
    """The files and coordinator metadata for one checkpoint image."""

    sharded: bool
    rank_files: tuple[RankCheckpointFile, ...]
    metadata_bytes: int

    @property
    def total_bytes(self) -> int:
        return sum(file.payload_bytes for file in self.rank_files) + self.metadata_bytes


@dataclass(frozen=True)
class CheckpointWriteEstimate:
    """Lower-bound write duration under per-rank and shared bandwidth limits."""

    rank_write_seconds: tuple[float, ...]
    slowest_rank_seconds: float
    shared_bandwidth_floor_seconds: float
    metadata_write_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class CheckpointRetention:
    """Whole checkpoint images that fit within a storage budget."""

    image_bytes: int
    storage_budget_bytes: int
    retained_images: int
    consumed_bytes: int
    remaining_bytes: int

    def covers_interval(self, interval_steps: int, required_history_steps: int) -> bool:
        """Return whether retained images cover a requested rollback history."""

        for name, value in (
            ("interval_steps", interval_steps),
            ("required_history_steps", required_history_steps),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise CheckpointPlanningError(f"{name} must be nonnegative")
        if required_history_steps == 0:
            return True
        if interval_steps == 0:
            return False
        return self.retained_images * interval_steps >= required_history_steps


def _positive_rate(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise CheckpointPlanningError(f"{name} must be finite and positive")
    return float(value)


def checkpoint_image(
    model: DecoderSpec,
    *,
    parameter_dtype: str,
    optimizer_dtype: str,
    optimizer_slots: int,
    world_size: int,
    sharded: bool,
    master_parameter_dtype: str | None = None,
    rank_local_bytes: int = 0,
    metadata_bytes: int = 0,
) -> CheckpointImage:
    """Build an exact checkpoint file inventory from declared state copies."""

    if not isinstance(model, DecoderSpec):
        raise TypeError("model must be a DecoderSpec")
    for name, value in (
        ("optimizer_slots", optimizer_slots),
        ("world_size", world_size),
        ("rank_local_bytes", rank_local_bytes),
        ("metadata_bytes", metadata_bytes),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if optimizer_slots < 0:
        raise CheckpointPlanningError("optimizer_slots must be nonnegative")
    if world_size <= 0:
        raise CheckpointPlanningError("world_size must be positive")
    if rank_local_bytes < 0 or metadata_bytes < 0:
        raise CheckpointPlanningError("checkpoint byte counts must be nonnegative")
    if not isinstance(sharded, bool):
        raise TypeError("sharded must be a boolean")

    parameter_count = model.parameter_count
    global_components = {
        "parameters": parameter_count * bytes_per_element(parameter_dtype),
        "optimizer": (
            parameter_count * bytes_per_element(optimizer_dtype) * optimizer_slots
        ),
    }
    if master_parameter_dtype is not None:
        global_components["master_parameters"] = parameter_count * bytes_per_element(
            master_parameter_dtype
        )

    rank_components = [[] for _ in range(world_size)]
    if sharded:
        for name, size in global_components.items():
            base, remainder = divmod(size, world_size)
            for rank in range(world_size):
                rank_components[rank].append((name, base + (rank < remainder)))
    else:
        rank_components[0].extend(global_components.items())
    for components in rank_components:
        if rank_local_bytes:
            components.append(("rank_local", rank_local_bytes))
    rank_files = tuple(
        RankCheckpointFile(rank, tuple(components))
        for rank, components in enumerate(rank_components)
    )
    return CheckpointImage(sharded, rank_files, metadata_bytes)


def checkpoint_write_estimate(
    image: CheckpointImage,
    *,
    rank_bandwidth_bytes_per_second: float,
    shared_bandwidth_bytes_per_second: float,
    metadata_bandwidth_bytes_per_second: float | None = None,
) -> CheckpointWriteEstimate:
    """Estimate parallel checkpoint writes without claiming measured throughput."""

    if not isinstance(image, CheckpointImage):
        raise TypeError("image must be a CheckpointImage")
    rank_rate = _positive_rate(
        rank_bandwidth_bytes_per_second, "rank_bandwidth_bytes_per_second"
    )
    shared_rate = _positive_rate(
        shared_bandwidth_bytes_per_second, "shared_bandwidth_bytes_per_second"
    )
    metadata_rate = (
        rank_rate
        if metadata_bandwidth_bytes_per_second is None
        else _positive_rate(
            metadata_bandwidth_bytes_per_second,
            "metadata_bandwidth_bytes_per_second",
        )
    )
    rank_seconds = tuple(file.payload_bytes / rank_rate for file in image.rank_files)
    slowest = max(rank_seconds)
    shared_floor = sum(file.payload_bytes for file in image.rank_files) / shared_rate
    metadata_seconds = image.metadata_bytes / metadata_rate
    return CheckpointWriteEstimate(
        rank_write_seconds=rank_seconds,
        slowest_rank_seconds=slowest,
        shared_bandwidth_floor_seconds=shared_floor,
        metadata_write_seconds=metadata_seconds,
        total_seconds=max(slowest, shared_floor) + metadata_seconds,
    )


def checkpoint_retention(
    image: CheckpointImage, *, storage_budget_bytes: int
) -> CheckpointRetention:
    """Compute whole-image retention without overcommitting storage."""

    if not isinstance(image, CheckpointImage):
        raise TypeError("image must be a CheckpointImage")
    if type(storage_budget_bytes) is not int:
        raise TypeError("storage_budget_bytes must be an integer")
    if storage_budget_bytes < 0:
        raise CheckpointPlanningError("storage_budget_bytes must be nonnegative")
    if image.total_bytes <= 0:
        raise CheckpointPlanningError("checkpoint image must contain bytes")
    retained = storage_budget_bytes // image.total_bytes
    consumed = retained * image.total_bytes
    return CheckpointRetention(
        image_bytes=image.total_bytes,
        storage_budget_bytes=storage_budget_bytes,
        retained_images=retained,
        consumed_bytes=consumed,
        remaining_bytes=storage_budget_bytes - consumed,
    )
