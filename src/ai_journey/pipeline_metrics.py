"""Analytical Pipeline Bubble evidence derived from validated schedules."""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline_schedule import PipelineSchedule, validate_pipeline_schedule


EVIDENCE_BOUNDARY = (
    "Analytical schedule-slot evidence only; no execution timing, hardware "
    "utilization, throughput, or production behavior was measured."
)


@dataclass(frozen=True)
class StageBubbleMetrics:
    """Busy and idle logical slots observed for one pipeline stage."""

    stage_index: int
    busy_slots: int
    idle_slots: int
    longest_idle_run: int

    def to_mapping(self) -> dict[str, int]:
        """Return the canonical public stage representation."""

        return {
            "stage_index": self.stage_index,
            "busy_slots": self.busy_slots,
            "idle_slots": self.idle_slots,
            "longest_idle_run": self.longest_idle_run,
        }


@dataclass(frozen=True)
class PipelineBubbleMetrics:
    """Schedule-derived utilization with its full stage-slot denominator."""

    policy: str
    stage_count: int
    microbatch_count: int
    schedule_slot_count: int
    busy_stage_slots: int
    idle_stage_slots: int
    stages: tuple[StageBubbleMetrics, ...]

    @property
    def stage_slot_denominator(self) -> int:
        """Return all observable stage positions across all schedule slots."""

        return self.stage_count * self.schedule_slot_count

    @property
    def bubble_fraction(self) -> float:
        """Return idle stage slots divided by all stage slots."""

        return self.idle_stage_slots / self.stage_slot_denominator

    @property
    def utilization(self) -> float:
        """Return busy stage slots divided by all stage slots."""

        return self.busy_stage_slots / self.stage_slot_denominator

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic evidence with explicit ratio denominators."""

        denominator = self.stage_slot_denominator
        return {
            "policy": self.policy,
            "stage_count": self.stage_count,
            "microbatch_count": self.microbatch_count,
            "schedule_slot_count": self.schedule_slot_count,
            "busy_stage_slots": self.busy_stage_slots,
            "idle_stage_slots": self.idle_stage_slots,
            "stage_slot_denominator": denominator,
            "bubble_fraction": {
                "numerator": self.idle_stage_slots,
                "denominator": denominator,
                "value": self.bubble_fraction,
            },
            "utilization": {
                "numerator": self.busy_stage_slots,
                "denominator": denominator,
                "value": self.utilization,
            },
            "stages": [stage.to_mapping() for stage in self.stages],
            "evidence_boundary": EVIDENCE_BOUNDARY,
        }


def measure_pipeline_bubbles(schedule: PipelineSchedule) -> PipelineBubbleMetrics:
    """Measure logical idle positions after fully validating a schedule."""

    if not isinstance(schedule, PipelineSchedule):
        raise TypeError("schedule must be a PipelineSchedule")
    validate_pipeline_schedule(schedule)
    if schedule.stage_count <= 0 or not schedule.slots:
        raise ValueError("pipeline schedule must have stage slots to measure")

    stages = []
    for stage_index in range(schedule.stage_count):
        occupied = tuple(
            any(operation.stage_index == stage_index for operation in slot.operations)
            for slot in schedule.slots
        )
        longest_idle_run = 0
        current_idle_run = 0
        for is_busy in occupied:
            if is_busy:
                current_idle_run = 0
            else:
                current_idle_run += 1
                longest_idle_run = max(longest_idle_run, current_idle_run)
        busy_slots = sum(occupied)
        stages.append(
            StageBubbleMetrics(
                stage_index=stage_index,
                busy_slots=busy_slots,
                idle_slots=len(schedule.slots) - busy_slots,
                longest_idle_run=longest_idle_run,
            )
        )

    busy_stage_slots = sum(stage.busy_slots for stage in stages)
    denominator = schedule.stage_count * len(schedule.slots)
    return PipelineBubbleMetrics(
        policy=schedule.policy,
        stage_count=schedule.stage_count,
        microbatch_count=schedule.microbatch_count,
        schedule_slot_count=len(schedule.slots),
        busy_stage_slots=busy_stage_slots,
        idle_stage_slots=denominator - busy_stage_slots,
        stages=tuple(stages),
    )
