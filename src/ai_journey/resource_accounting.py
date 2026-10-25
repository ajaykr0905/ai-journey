"""Deterministic resource accounting for dense decoder-only transformers.

The formulas in this module are planning estimates.  They do not claim measured
GPU allocation, throughput, or kernel behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


class ResourceAccountingError(ValueError):
    """Raised when a resource-accounting contract is invalid."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ResourceAccountingError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class DecoderSpec:
    """Architecture inputs for a dense SwiGLU decoder-only transformer."""

    vocab_size: int
    layer_count: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    feed_forward_size: int
    tie_embeddings: bool = False
    use_bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "layer_count",
            "hidden_size",
            "attention_heads",
            "kv_heads",
            "feed_forward_size",
        ):
            _positive_integer(getattr(self, name), name)
        if self.hidden_size % self.attention_heads:
            raise ResourceAccountingError(
                "hidden_size must be divisible by attention_heads"
            )
        if self.attention_heads % self.kv_heads:
            raise ResourceAccountingError(
                "attention_heads must be divisible by kv_heads"
            )
        if not isinstance(self.tie_embeddings, bool):
            raise TypeError("tie_embeddings must be a boolean")
        if not isinstance(self.use_bias, bool):
            raise TypeError("use_bias must be a boolean")

    @property
    def head_size(self) -> int:
        """Return the width of one attention head."""

        return self.hidden_size // self.attention_heads

    @property
    def kv_size(self) -> int:
        """Return the combined key/value projection width."""

        return self.kv_heads * self.head_size

    def parameter_groups(self) -> dict[str, int]:
        """Return an exact, non-overlapping parameter inventory."""

        hidden = self.hidden_size
        kv_size = self.kv_size
        attention = hidden * hidden * 2 + hidden * kv_size * 2
        feed_forward = hidden * self.feed_forward_size * 3
        norms = hidden * 2
        if self.use_bias:
            attention += hidden * 2 + kv_size * 2
            feed_forward += self.feed_forward_size * 2 + hidden
        groups = {
            "token_embedding": self.vocab_size * hidden,
            "attention": self.layer_count * attention,
            "feed_forward": self.layer_count * feed_forward,
            "block_norms": self.layer_count * norms,
            "final_norm": hidden,
            "output_projection": (
                0 if self.tie_embeddings else self.vocab_size * hidden
            ),
        }
        return groups

    @property
    def parameter_count(self) -> int:
        """Return the exact parameter count for the declared architecture."""

        return sum(self.parameter_groups().values())


@dataclass(frozen=True)
class TrainingWorkload:
    """The token geometry of one data-parallel training run."""

    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    data_parallel_size: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        for name in (
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "data_parallel_size",
            "optimizer_steps",
        ):
            _positive_integer(getattr(self, name), name)

    @property
    def tokens_per_micro_batch(self) -> int:
        return self.sequence_length * self.micro_batch_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        return (
            self.tokens_per_micro_batch
            * self.gradient_accumulation_steps
            * self.data_parallel_size
        )

    @property
    def total_training_tokens(self) -> int:
        return self.tokens_per_optimizer_step * self.optimizer_steps


def forward_linear_flops(
    model: DecoderSpec, *, batch_size: int, sequence_length: int
) -> dict[str, int]:
    """Count multiply-adds as two FLOPs for all dense forward projections."""

    _positive_integer(batch_size, "batch_size")
    _positive_integer(sequence_length, "sequence_length")
    tokens = batch_size * sequence_length
    hidden = model.hidden_size
    kv_size = model.kv_size
    per_layer_attention_weights = hidden * hidden * 2 + hidden * kv_size * 2
    per_layer_feed_forward_weights = hidden * model.feed_forward_size * 3
    return {
        "attention_projections": (
            2 * tokens * model.layer_count * per_layer_attention_weights
        ),
        "feed_forward_projections": (
            2 * tokens * model.layer_count * per_layer_feed_forward_weights
        ),
        "output_projection": 2 * tokens * hidden * model.vocab_size,
    }
