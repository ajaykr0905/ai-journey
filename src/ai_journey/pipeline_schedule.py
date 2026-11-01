"""Dependency-valid logical schedules for pipeline-parallel execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pipeline_partition import PipelinePartitionPlan


PipelineDirection = Literal["forward", "backward"]
PipelinePhase = Literal["forward", "backward", "warmup", "steady", "cooldown"]
PipelinePolicy = Literal["gpipe", "1f1b"]


@dataclass(frozen=True)
class PipelineOperation:
    """One stage's work for one microbatch in a logical schedule slot."""

    stage_index: int
    microbatch_index: int
    direction: PipelineDirection
    phase: PipelinePhase
    dependencies: tuple[str, ...]

    @property
    def identity(self) -> str:
        """Return the canonical identity used by dependent operations."""

        return f"{self.direction}:{self.stage_index}:{self.microbatch_index}"

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical public operation representation."""

        return {
            "identity": self.identity,
            "stage_index": self.stage_index,
            "microbatch_index": self.microbatch_index,
            "direction": self.direction,
            "phase": self.phase,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True)
class PipelineSlot:
    """Operations that may execute concurrently in one logical time step."""

    index: int
    operations: tuple[PipelineOperation, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical public slot representation."""

        return {
            "index": self.index,
            "operations": [operation.to_mapping() for operation in self.operations],
        }


@dataclass(frozen=True)
class PipelineSchedule:
    """An immutable pipeline policy and its dependency-ordered slots."""

    policy: PipelinePolicy
    stage_count: int
    microbatch_count: int
    slots: tuple[PipelineSlot, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic values suitable for replay and reporting."""

        return {
            "policy": self.policy,
            "stage_count": self.stage_count,
            "microbatch_count": self.microbatch_count,
            "slots": [slot.to_mapping() for slot in self.slots],
        }


def validate_pipeline_schedule(schedule: PipelineSchedule) -> None:
    """Reject schedules that do not contain each required operation once."""

    if schedule.policy not in ("gpipe", "1f1b"):
        raise ValueError("pipeline schedule policy must be gpipe or 1f1b")
    if tuple(slot.index for slot in schedule.slots) != tuple(
        range(len(schedule.slots))
    ):
        raise ValueError("pipeline schedule slot indices must be consecutive")
    expected = {
        f"{direction}:{stage_index}:{microbatch_index}"
        for direction in ("forward", "backward")
        for stage_index in range(schedule.stage_count)
        for microbatch_index in range(schedule.microbatch_count)
    }
    identities = [
        operation.identity for slot in schedule.slots for operation in slot.operations
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("pipeline schedule contains duplicate operations")
    if set(identities) != expected:
        raise ValueError("pipeline schedule has missing or unexpected operations")
    completed: set[str] = set()
    backward_started = False
    for slot in schedule.slots:
        stage_indices = [operation.stage_index for operation in slot.operations]
        if len(stage_indices) != len(set(stage_indices)):
            raise ValueError("pipeline slots allow one operation per stage")
        directions = {operation.direction for operation in slot.operations}
        if schedule.policy == "gpipe" and "forward" in directions:
            if backward_started or "backward" in directions:
                raise ValueError("GPipe forwards must finish before backwards start")
        backward_started = backward_started or "backward" in directions
        for operation in slot.operations:
            expected_operation = _operation(
                operation.direction,
                operation.stage_index,
                operation.microbatch_index,
                schedule.stage_count,
            )
            if operation.dependencies != expected_operation.dependencies:
                raise ValueError(f"dependencies do not match for {operation.identity}")
            if schedule.policy == "gpipe" and operation.phase != operation.direction:
                raise ValueError(f"phase does not match for {operation.identity}")
            if not set(operation.dependencies).issubset(completed):
                raise ValueError(
                    f"dependency must precede operation {operation.identity}"
                )
        completed.update(operation.identity for operation in slot.operations)


def _operation(
    direction: PipelineDirection,
    stage_index: int,
    microbatch_index: int,
    stage_count: int,
) -> PipelineOperation:
    if direction == "forward":
        dependencies = (
            ()
            if stage_index == 0
            else (f"forward:{stage_index - 1}:{microbatch_index}",)
        )
        phase: PipelinePhase = "forward"
    else:
        dependencies = (f"forward:{stage_index}:{microbatch_index}",)
        if stage_index < stage_count - 1:
            dependencies += (f"backward:{stage_index + 1}:{microbatch_index}",)
        phase = "backward"
    return PipelineOperation(
        stage_index=stage_index,
        microbatch_index=microbatch_index,
        direction=direction,
        phase=phase,
        dependencies=dependencies,
    )


def generate_gpipe_schedule(
    stage_plan: PipelinePartitionPlan, *, microbatch_count: int
) -> PipelineSchedule:
    """Generate GPipe's forward drain followed by reverse backward drain."""

    if not isinstance(stage_plan, PipelinePartitionPlan):
        raise TypeError("stage_plan must be a PipelinePartitionPlan")
    if type(microbatch_count) is not int:
        raise TypeError("microbatch_count must be an integer")
    if microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive")
    stage_count = len(stage_plan.stages)
    slots: list[PipelineSlot] = []
    for step in range(stage_count + microbatch_count - 1):
        operations = []
        for stage_index in range(stage_count):
            microbatch_index = step - stage_index
            if 0 <= microbatch_index < microbatch_count:
                operations.append(
                    _operation("forward", stage_index, microbatch_index, stage_count)
                )
        slots.append(PipelineSlot(index=len(slots), operations=tuple(operations)))
    for step in range(stage_count + microbatch_count - 1):
        operations = []
        for stage_index in range(stage_count):
            microbatch_index = (
                microbatch_count - 1 - step + (stage_count - 1 - stage_index)
            )
            if 0 <= microbatch_index < microbatch_count:
                operations.append(
                    _operation("backward", stage_index, microbatch_index, stage_count)
                )
        slots.append(PipelineSlot(index=len(slots), operations=tuple(operations)))
    schedule = PipelineSchedule(
        policy="gpipe",
        stage_count=stage_count,
        microbatch_count=microbatch_count,
        slots=tuple(slots),
    )
    validate_pipeline_schedule(schedule)
    return schedule
