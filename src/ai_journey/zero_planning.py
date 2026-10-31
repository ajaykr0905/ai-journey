"""Byte-exact planning primitives for ZeRO data-parallel stages."""

from __future__ import annotations

from dataclasses import dataclass

from .fsdp_planning import partition_parameter_groups
from .resource_accounting import DecoderSpec, bytes_per_element


class ZeroPlanningError(ValueError):
    """Raised when a ZeRO planning contract is invalid."""


_SHARDED_FAMILIES = {
    0: frozenset(),
    1: frozenset({"optimizer", "master_parameters"}),
    2: frozenset({"gradients", "optimizer", "master_parameters"}),
    3: frozenset({"parameters", "gradients", "optimizer", "master_parameters"}),
}


@dataclass(frozen=True)
class RankZeroState:
    """Persistent training-state ownership for one data-parallel rank."""

    rank: int
    stage: int
    parameters_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    master_parameters_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return the complete persistent state held by this rank."""

        return (
            self.parameters_bytes
            + self.gradients_bytes
            + self.optimizer_bytes
            + self.master_parameters_bytes
        )


def zero_stage_state(
    model: DecoderSpec,
    *,
    stage: int,
    world_size: int,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_dtype: str,
    optimizer_slots: int,
    master_parameter_dtype: str | None = None,
) -> tuple[RankZeroState, ...]:
    """Inventory exact persistent bytes for ZeRO stages 0 through 3."""

    if not isinstance(model, DecoderSpec):
        raise TypeError("model must be a DecoderSpec")
    if type(stage) is not int:
        raise TypeError("stage must be an integer")
    if stage not in _SHARDED_FAMILIES:
        raise ZeroPlanningError("stage must be one of 0, 1, 2, or 3")
    if type(optimizer_slots) is not int:
        raise TypeError("optimizer_slots must be an integer")
    if optimizer_slots < 0:
        raise ZeroPlanningError("optimizer_slots must be nonnegative")

    parameter_width = bytes_per_element(parameter_dtype)
    gradient_width = bytes_per_element(gradient_dtype)
    optimizer_width = bytes_per_element(optimizer_dtype) * optimizer_slots
    master_width = (
        0
        if master_parameter_dtype is None
        else bytes_per_element(master_parameter_dtype)
    )
    parameter_count = model.parameter_count
    count_shards = partition_parameter_groups(
        model.parameter_groups(), world_size=world_size
    )
    sharded = _SHARDED_FAMILIES[stage]

    def owned_bytes(family: str, shard_count: int, width: int) -> int:
        count = shard_count if family in sharded else parameter_count
        return count * width

    return tuple(
        RankZeroState(
            rank=shard.rank,
            stage=stage,
            parameters_bytes=owned_bytes(
                "parameters", shard.payload_bytes, parameter_width
            ),
            gradients_bytes=owned_bytes(
                "gradients", shard.payload_bytes, gradient_width
            ),
            optimizer_bytes=owned_bytes(
                "optimizer", shard.payload_bytes, optimizer_width
            ),
            master_parameters_bytes=owned_bytes(
                "master_parameters", shard.payload_bytes, master_width
            ),
        )
        for shard in count_shards
    )
