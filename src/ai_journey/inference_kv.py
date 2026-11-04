"""Exact analytical KV-cache accounting for decoder-only inference."""

from __future__ import annotations

from dataclasses import dataclass


class InferenceCapacityError(ValueError):
    """Raised when an inference-capacity contract is invalid."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise InferenceCapacityError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class KVCacheLayout:
    """Shape and precision controls for a decoder KV cache."""

    layers: int
    kv_heads: int
    head_dim: int
    context_tokens: int
    batch_size: int
    bytes_per_element: int = 2

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "kv_heads",
            "head_dim",
            "context_tokens",
            "batch_size",
            "bytes_per_element",
        ):
            _positive_integer(getattr(self, name), name)

    @property
    def bytes_per_token_per_sequence(self) -> int:
        """Return key plus value bytes for one token in one sequence."""

        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element

    @property
    def total_bytes(self) -> int:
        """Return the dense cache allocation at the declared batch/context."""

        return self.bytes_per_token_per_sequence * self.context_tokens * self.batch_size
