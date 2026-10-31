"""Byte-exact planning primitives for ZeRO data-parallel stages."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .distributed_capacity import (
    ClusterTopology,
    ring_all_gather,
    ring_all_reduce,
    ring_reduce_scatter,
)
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


@dataclass(frozen=True)
class ZeroCollectiveStep:
    """Logical tensor ownership before and after one ZeRO collective."""

    phase: str
    operation: str
    tensor_family: str
    input_bytes_by_rank: tuple[int, ...]
    output_bytes_by_rank: tuple[int, ...]


@dataclass(frozen=True)
class RankZeroMemoryPeak:
    """Analytical persistent and non-overlapping transient memory for one rank."""

    rank: int
    stage: int
    persistent_bytes: int
    activation_bytes: int
    parameter_materialization_bytes: int
    gradient_partition_buffer_bytes: int
    peak_bytes: int
    dominant_transient: str


@dataclass(frozen=True)
class ZeroCollectiveEstimate:
    """Ring cost for one logical ZeRO collective, including padding."""

    phase: str
    operation: str
    tensor_family: str
    logical_payload_bytes: int
    padded_payload_bytes: int
    padding_bytes: int
    wire_bytes_per_rank: int
    seconds: float
    link_scope: str


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


def zero_collective_schedule(
    model: DecoderSpec,
    *,
    stage: int,
    world_size: int,
    parameter_dtype: str,
    gradient_dtype: str,
) -> tuple[ZeroCollectiveStep, ...]:
    """Describe logical collective transitions for one ZeRO optimizer step."""

    if not isinstance(model, DecoderSpec):
        raise TypeError("model must be a DecoderSpec")
    if type(stage) is not int:
        raise TypeError("stage must be an integer")
    if stage not in _SHARDED_FAMILIES:
        raise ZeroPlanningError("stage must be one of 0, 1, 2, or 3")

    count_shards = partition_parameter_groups(
        model.parameter_groups(), world_size=world_size
    )
    parameter_width = bytes_per_element(parameter_dtype)
    gradient_width = bytes_per_element(gradient_dtype)
    full_parameter_bytes = model.parameter_count * parameter_width
    full_gradient_bytes = model.parameter_count * gradient_width
    parameter_shards = tuple(
        shard.payload_bytes * parameter_width for shard in count_shards
    )
    gradient_shards = tuple(
        shard.payload_bytes * gradient_width for shard in count_shards
    )
    full_parameters = (full_parameter_bytes,) * world_size
    full_gradients = (full_gradient_bytes,) * world_size

    gradient_all_reduce = ZeroCollectiveStep(
        "gradient_sync",
        "all_reduce",
        "gradients",
        full_gradients,
        full_gradients,
    )
    gradient_reduce_scatter = ZeroCollectiveStep(
        "gradient_partition",
        "reduce_scatter",
        "gradients",
        full_gradients,
        gradient_shards,
    )
    parameter_sync = ZeroCollectiveStep(
        "parameter_sync",
        "all_gather",
        "parameters",
        parameter_shards,
        full_parameters,
    )
    parameter_materialization = ZeroCollectiveStep(
        "parameter_materialization",
        "all_gather",
        "parameters",
        parameter_shards,
        full_parameters,
    )
    return {
        0: (gradient_all_reduce,),
        1: (gradient_all_reduce, parameter_sync),
        2: (gradient_reduce_scatter, parameter_sync),
        3: (parameter_materialization, gradient_reduce_scatter),
    }[stage]


def zero_stage_memory_peaks(
    model: DecoderSpec,
    *,
    stage: int,
    world_size: int,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_dtype: str,
    optimizer_slots: int,
    activation_bytes: int,
    gradient_bucket_bytes: int,
    master_parameter_dtype: str | None = None,
) -> tuple[RankZeroMemoryPeak, ...]:
    """Estimate rank peaks when parameter and gradient transients do not overlap."""

    if type(activation_bytes) is not int:
        raise TypeError("activation_bytes must be an integer")
    if activation_bytes < 0:
        raise ZeroPlanningError("activation_bytes must be nonnegative")
    if type(gradient_bucket_bytes) is not int:
        raise TypeError("gradient_bucket_bytes must be an integer")
    if gradient_bucket_bytes <= 0:
        raise ZeroPlanningError("gradient_bucket_bytes must be positive")

    states = zero_stage_state(
        model,
        stage=stage,
        world_size=world_size,
        parameter_dtype=parameter_dtype,
        gradient_dtype=gradient_dtype,
        optimizer_dtype=optimizer_dtype,
        optimizer_slots=optimizer_slots,
        master_parameter_dtype=master_parameter_dtype,
    )
    full_parameter_bytes = model.parameter_count * bytes_per_element(parameter_dtype)
    full_gradient_bytes = model.parameter_count * bytes_per_element(gradient_dtype)
    peaks = []
    for state in states:
        parameter_transient = (
            full_parameter_bytes - state.parameters_bytes if stage == 3 else 0
        )
        gradient_transient = (
            min(full_gradient_bytes - state.gradients_bytes, gradient_bucket_bytes)
            if stage >= 2
            else 0
        )
        transient = max(parameter_transient, gradient_transient)
        if not transient:
            dominant = "none"
        elif parameter_transient >= gradient_transient:
            dominant = "parameter_materialization"
        else:
            dominant = "gradient_partition"
        peaks.append(
            RankZeroMemoryPeak(
                rank=state.rank,
                stage=stage,
                persistent_bytes=state.total_bytes,
                activation_bytes=activation_bytes,
                parameter_materialization_bytes=parameter_transient,
                gradient_partition_buffer_bytes=gradient_transient,
                peak_bytes=state.total_bytes + activation_bytes + transient,
                dominant_transient=dominant,
            )
        )
    return tuple(peaks)


def estimate_zero_collectives(
    model: DecoderSpec,
    *,
    stage: int,
    topology: ClusterTopology,
    parameter_dtype: str,
    gradient_dtype: str,
) -> tuple[ZeroCollectiveEstimate, ...]:
    """Estimate flat-ring wire work for a stage's logical collective schedule."""

    if not isinstance(topology, ClusterTopology):
        raise TypeError("topology must be a ClusterTopology")
    schedule = zero_collective_schedule(
        model,
        stage=stage,
        world_size=topology.world_size,
        parameter_dtype=parameter_dtype,
        gradient_dtype=gradient_dtype,
    )
    estimates = []
    for step in schedule:
        if step.operation == "all_gather":
            shard_bytes = max(step.input_bytes_by_rank)
            logical_payload = sum(step.input_bytes_by_rank)
            padded_payload = shard_bytes * topology.world_size
            ring = ring_all_gather(shard_bytes, topology)
        else:
            logical_payload = step.input_bytes_by_rank[0]
            padded_payload = (
                math.ceil(logical_payload / topology.world_size) * topology.world_size
            )
            if step.operation == "all_reduce":
                ring = ring_all_reduce(logical_payload, topology)
            else:
                ring = ring_reduce_scatter(logical_payload, topology)
        estimates.append(
            ZeroCollectiveEstimate(
                phase=step.phase,
                operation=step.operation,
                tensor_family=step.tensor_family,
                logical_payload_bytes=logical_payload,
                padded_payload_bytes=padded_payload,
                padding_bytes=padded_payload - logical_payload,
                wire_bytes_per_rank=ring.wire_bytes_per_rank,
                seconds=ring.seconds,
                link_scope=ring.link_scope,
            )
        )
    return tuple(estimates)
