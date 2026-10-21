"""Deterministic GPT-2 training controls with exact restart evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math


class GPT2TrainingError(ValueError):
    """Raised when a GPT-2 training contract or run is invalid."""


@dataclass(frozen=True)
class GPT2TrainingConfig:
    """Validated optimizer, accumulation, schedule, and reproducibility controls."""

    optimizer_steps: int
    micro_batch_size: int
    accumulation_steps: int
    sequence_length: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "optimizer_steps",
            "micro_batch_size",
            "accumulation_steps",
            "sequence_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise GPT2TrainingError(f"{name} must be positive")
        if isinstance(self.warmup_steps, bool) or not isinstance(
            self.warmup_steps, int
        ):
            raise TypeError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps <= self.optimizer_steps:
            raise GPT2TrainingError(
                "warmup_steps must be between zero and optimizer_steps"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**32:
            raise GPT2TrainingError("seed must be in [0, 2**32)")
        for name in (
            "max_learning_rate",
            "min_learning_rate",
            "weight_decay",
            "gradient_clip",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise GPT2TrainingError(f"{name} must be finite")
        if self.max_learning_rate <= 0:
            raise GPT2TrainingError("max_learning_rate must be positive")
        if not 0 <= self.min_learning_rate <= self.max_learning_rate:
            raise GPT2TrainingError(
                "min_learning_rate must be between zero and max_learning_rate"
            )
        if self.weight_decay < 0:
            raise GPT2TrainingError("weight_decay must be non-negative")
        if self.gradient_clip <= 0:
            raise GPT2TrainingError("gradient_clip must be positive")

    @property
    def effective_batch_size(self) -> int:
        """Return examples contributing to each optimizer update."""

        return self.micro_batch_size * self.accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        """Return target-token exposure for one optimizer update."""

        return self.effective_batch_size * self.sequence_length

    @property
    def total_target_tokens(self) -> int:
        """Return target-token exposure for the complete schedule."""

        return self.tokens_per_step * self.optimizer_steps

    def fingerprint(self) -> str:
        """Return a canonical digest of every training control."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()
