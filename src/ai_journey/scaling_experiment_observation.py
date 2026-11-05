"""Measured observations for bounded scaling-law experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .scaling_experiment_identity import ExperimentInputs, run_identity
from .scaling_experiment_protocol import ExperimentPoint
from .scaling_laws import ScalingLawError, _positive_integer


@dataclass(frozen=True)
class ScalingObservation:
    run_id: str
    point: ExperimentPoint
    validation_loss: float
    trained_tokens: int
    optimizer_steps: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or len(self.run_id) != 64
            or any(character not in "0123456789abcdef" for character in self.run_id)
        ):
            raise ScalingLawError("run_id must be a lowercase SHA-256 digest")
        if not isinstance(self.point, ExperimentPoint):
            raise TypeError("point must be ExperimentPoint")
        if (
            isinstance(self.validation_loss, bool)
            or not isinstance(self.validation_loss, (int, float))
            or not math.isfinite(self.validation_loss)
            or self.validation_loss <= 0
        ):
            raise ScalingLawError("validation_loss must be positive and finite")
        _positive_integer(self.trained_tokens, "trained_tokens")
        _positive_integer(self.optimizer_steps, "optimizer_steps")
        if self.trained_tokens != self.point.train_tokens:
            raise ScalingLawError("trained_tokens must equal the predeclared target")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds <= 0
        ):
            raise ScalingLawError("elapsed_seconds must be positive and finite")


def verify_observation_identity(
    observation: ScalingObservation, inputs: ExperimentInputs
) -> ScalingObservation:
    """Reject an observation whose run digest does not bind its inputs."""

    if not isinstance(observation, ScalingObservation):
        raise TypeError("observation must be ScalingObservation")
    expected = run_identity(observation.point, inputs)
    if observation.run_id != expected:
        raise ScalingLawError("observation run identity does not match its inputs")
    return observation
