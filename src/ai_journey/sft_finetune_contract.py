"""Validated controls for bounded public-data SFT experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


class SFTFineTuneError(ValueError):
    """The fine-tuning input or evidence violates its declared contract."""


@dataclass(frozen=True)
class SFTFineTuneConfig:
    """All architecture, optimization, and evidence limits for one run."""

    seed: int = 68
    block_size: int = 64
    embedding_dim: int = 16
    head_count: int = 2
    layer_count: int = 1
    batch_size: int = 4
    steps: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    evaluation_examples: int = 8
    generation_tokens: int = 16
    checkpoint_interval: int = 4

    def __post_init__(self) -> None:
        for name, low, high in (
            ("seed", 0, 2**32 - 1),
            ("block_size", 8, 256),
            ("embedding_dim", 4, 64),
            ("head_count", 1, 8),
            ("layer_count", 1, 4),
            ("batch_size", 1, 16),
            ("steps", 1, 128),
            ("evaluation_examples", 1, 50),
            ("generation_tokens", 1, 64),
            ("checkpoint_interval", 1, 128),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not low <= value <= high:
                raise SFTFineTuneError(f"{name} must be in [{low}, {high}]")
        if self.embedding_dim % self.head_count:
            raise SFTFineTuneError("embedding_dim must be divisible by head_count")
        if self.checkpoint_interval > self.steps:
            raise SFTFineTuneError("checkpoint_interval cannot exceed steps")
        for name, low, high, inclusive_low in (
            ("learning_rate", 0.0, 0.1, False),
            ("weight_decay", 0.0, 1.0, True),
            ("gradient_clip", 0.0, 100.0, False),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not (low <= value <= high if inclusive_low else low < value <= high)
            ):
                raise SFTFineTuneError(f"{name} must be finite and in ({low}, {high}]")
        if self.work_units > 100_000_000:
            raise SFTFineTuneError("fine-tuning plan exceeds the CPU work budget")

    @property
    def work_units(self) -> int:
        return (
            self.steps
            * self.batch_size
            * self.block_size
            * self.embedding_dim**2
            * self.layer_count
        )

    def as_mapping(self) -> dict:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> "SFTFineTuneConfig":
        expected = {field.name for field in fields(cls)}
        if not isinstance(value, dict) or set(value) != expected:
            raise SFTFineTuneError("fine-tuning config requires exactly all controls")
        return cls(**value)
