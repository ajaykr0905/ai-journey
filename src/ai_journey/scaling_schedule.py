"""Align analytical token targets with executable optimizer-step geometry."""

from __future__ import annotations

from dataclasses import dataclass

from .scaling_laws import ScalingLawError, _positive_integer


@dataclass(frozen=True)
class TrainingBatchGeometry:
    """Sequence, micro-batch, accumulation and data-parallel sizes of one update.

    All four must be positive integers; together they set the token granularity
    that token targets are rounded to.
    """

    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    data_parallel_size: int

    def __post_init__(self) -> None:
        for name in (
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_size",
        ):
            _positive_integer(getattr(self, name), name)

    @property
    def tokens_per_optimizer_step(self) -> int:
        """Return tokens consumed by one globally synchronized optimizer step."""

        return (
            self.sequence_length
            * self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.data_parallel_size
        )


@dataclass(frozen=True)
class TokenSchedule:
    """A token target rounded up to whole optimizer steps, overshoot explicit."""

    target_tokens: int
    optimizer_steps: int
    scheduled_tokens: int
    overshoot_tokens: int
    geometry: TrainingBatchGeometry

    @property
    def exact(self) -> bool:
        """Return True when the target is a whole number of optimizer steps."""

        return self.overshoot_tokens == 0


def align_token_target(
    target_tokens: int, geometry: TrainingBatchGeometry
) -> TokenSchedule:
    """Round a token target to complete, globally synchronized updates."""

    target = _positive_integer(target_tokens, "target_tokens")
    if not isinstance(geometry, TrainingBatchGeometry):
        raise TypeError("geometry must be TrainingBatchGeometry")
    per_step = geometry.tokens_per_optimizer_step
    steps = (target + per_step - 1) // per_step
    scheduled = steps * per_step
    return TokenSchedule(
        target_tokens=target,
        optimizer_steps=steps,
        scheduled_tokens=scheduled,
        overshoot_tokens=scheduled - target,
        geometry=geometry,
    )


def validate_overshoot(
    schedule: TokenSchedule, *, maximum_fraction: float
) -> TokenSchedule:
    """Reject a schedule whose batch alignment materially changes the target."""

    if not isinstance(schedule, TokenSchedule):
        raise TypeError("schedule must be TokenSchedule")
    if (
        isinstance(maximum_fraction, bool)
        or not isinstance(maximum_fraction, (int, float))
        or not 0 <= maximum_fraction < 1
    ):
        raise ScalingLawError("maximum_fraction must be in [0, 1)")
    if schedule.overshoot_tokens / schedule.target_tokens > maximum_fraction:
        raise ScalingLawError("batch-alignment overshoot exceeds the declared limit")
    return schedule
