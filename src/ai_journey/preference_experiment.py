"""Matched full and low-rank preference optimization on a tiny CPU decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math

from .preference_data import PreferenceDataError
from .transformer_lab import TransformerConfig


@dataclass(frozen=True)
class PreferenceComparisonConfig:
    seeds: tuple[int, ...] = (44, 45)
    fold_count: int = 3
    fold_seed: int = 4403
    block_size: int = 32
    embedding_dim: int = 16
    head_count: int = 2
    sft_batch_size: int = 4
    sft_steps: int = 4
    preference_batch_size: int = 2
    preference_steps: int = 4
    sft_learning_rate: float = 0.01
    preference_learning_rate: float = 0.01
    beta: float = 0.2
    adapter_scale: float = 2.0

    def __post_init__(self):
        if not isinstance(self.seeds, tuple) or not 2 <= len(self.seeds) <= 4:
            raise PreferenceDataError("seeds require two to four paired values")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise PreferenceDataError("seeds must be integers in [0, 2**32)")
        if len(set(self.seeds)) != len(self.seeds):
            raise PreferenceDataError("seeds must be distinct")
        for name, low, high in (
            ("fold_count", 2, 6),
            ("fold_seed", 0, 2**32 - 1),
            ("block_size", 8, 128),
            ("embedding_dim", 4, 32),
            ("head_count", 1, 8),
            ("sft_batch_size", 1, 16),
            ("sft_steps", 1, 32),
            ("preference_batch_size", 1, 8),
            ("preference_steps", 1, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise PreferenceDataError(
                    f"{name} must be an integer in [{low}, {high}]"
                )
        if self.embedding_dim % self.head_count:
            raise PreferenceDataError("embedding_dim must divide evenly across heads")
        for name, high in (
            ("sft_learning_rate", 0.1),
            ("preference_learning_rate", 0.1),
            ("beta", 10.0),
            ("adapter_scale", 8.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= high
            ):
                raise PreferenceDataError(f"{name} must be finite and in (0, {high}]")
        if self.work_units > 80_000_000:
            raise PreferenceDataError(
                "preference comparison exceeds the CPU work budget"
            )

    @property
    def work_units(self):
        return (
            len(self.seeds)
            * (self.sft_steps + self.fold_count * 4 * self.preference_steps)
            * max(self.sft_batch_size, self.preference_batch_size)
            * self.block_size
            * self.embedding_dim**2
        )

    def model_config(self, vocab_size):
        return TransformerConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            embedding_dim=self.embedding_dim,
            head_count=self.head_count,
            layer_count=1,
            feed_forward_expansion=2,
            dropout=0.0,
        )


def preference_config_dict(config):
    if not isinstance(config, PreferenceComparisonConfig):
        raise TypeError("config must be PreferenceComparisonConfig")
    return {**asdict(config), "seeds": list(config.seeds)}


def preference_config_from_dict(value):
    if (
        not isinstance(value, dict)
        or set(value) != {field.name for field in fields(PreferenceComparisonConfig)}
        or not isinstance(value["seeds"], list)
    ):
        raise PreferenceDataError(
            "config requires exactly all preference controls and a seeds list"
        )
    return PreferenceComparisonConfig(**{**value, "seeds": tuple(value["seeds"])})
