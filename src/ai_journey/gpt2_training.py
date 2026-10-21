"""Deterministic GPT-2 training controls with exact restart evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import torch
from torch import Tensor


class GPT2TrainingError(ValueError):
    """Raised when a GPT-2 training contract or run is invalid."""


@dataclass(frozen=True)
class GPT2ByteCorpus:
    """Immutable byte-token corpus with exact source provenance."""

    tokens: Tensor
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, Tensor)
            or self.tokens.layout != torch.strided
            or self.tokens.device.type != "cpu"
            or self.tokens.dtype != torch.long
            or self.tokens.ndim != 1
        ):
            raise TypeError("tokens must be a dense one-dimensional CPU long tensor")
        if self.tokens.numel() < 2:
            raise GPT2TrainingError("byte corpus requires at least two tokens")
        if bool(((self.tokens < 0) | (self.tokens > 255)).any()):
            raise GPT2TrainingError("byte corpus token ids must be in [0, 255]")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_sha256
            )
        ):
            raise GPT2TrainingError("source_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "tokens", self.tokens.detach().clone())

    @classmethod
    def from_bytes(
        cls, source: bytes, *, max_bytes: int = 16 * 1024 * 1024
    ) -> "GPT2ByteCorpus":
        """Build a byte-token corpus under an explicit input-size budget."""

        if not isinstance(source, bytes):
            raise TypeError("source must be bytes")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise GPT2TrainingError("max_bytes must be positive")
        if len(source) < 2:
            raise GPT2TrainingError("source must contain at least two bytes")
        if len(source) > max_bytes:
            raise GPT2TrainingError("source exceeds max_bytes")
        return cls(
            tokens=torch.tensor(tuple(source), dtype=torch.long),
            source_sha256=sha256(source).hexdigest(),
        )

    def fingerprint(self) -> str:
        """Bind the declared source digest to the materialized token sequence."""

        digest = sha256()
        digest.update(self.source_sha256.encode())
        digest.update(self.tokens.numpy().tobytes())
        return digest.hexdigest()


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
