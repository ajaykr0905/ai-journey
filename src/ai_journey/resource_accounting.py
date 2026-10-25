"""Deterministic resource accounting for dense decoder-only transformers.

The formulas in this module are planning estimates.  They do not claim measured
GPU allocation, throughput, or kernel behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class ResourceAccountingError(ValueError):
    """Raised when a resource-accounting contract is invalid."""


DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int8": 1,
}


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


def attention_matmul_flops(
    model: DecoderSpec, *, batch_size: int, sequence_length: int
) -> dict[str, int]:
    """Count dense QKᵀ and probability-value matmuls.

    Causal masking does not reduce this analytical count because standard dense
    kernels still materialize the full square matmul work.
    """

    _positive_integer(batch_size, "batch_size")
    _positive_integer(sequence_length, "sequence_length")
    one_matmul = (
        2 * batch_size * model.layer_count * sequence_length**2 * model.hidden_size
    )
    return {
        "query_key_scores": one_matmul,
        "attention_value_mix": one_matmul,
    }


def forward_flops(
    model: DecoderSpec, *, batch_size: int, sequence_length: int
) -> dict[str, int]:
    """Return the complete modeled forward-pass FLOP decomposition."""

    return {
        **forward_linear_flops(
            model, batch_size=batch_size, sequence_length=sequence_length
        ),
        **attention_matmul_flops(
            model, batch_size=batch_size, sequence_length=sequence_length
        ),
    }


def training_step_flops(
    model: DecoderSpec,
    workload: TrainingWorkload,
    *,
    backward_to_forward_ratio: float = 2.0,
) -> dict[str, int]:
    """Estimate aggregate forward/backward FLOPs for one optimizer step."""

    if (
        isinstance(backward_to_forward_ratio, bool)
        or not isinstance(backward_to_forward_ratio, (int, float))
        or not math.isfinite(backward_to_forward_ratio)
        or backward_to_forward_ratio <= 0
    ):
        raise ResourceAccountingError(
            "backward_to_forward_ratio must be positive and finite"
        )
    per_micro_batch = sum(
        forward_flops(
            model,
            batch_size=workload.micro_batch_size,
            sequence_length=workload.sequence_length,
        ).values()
    )
    forward = (
        per_micro_batch
        * workload.gradient_accumulation_steps
        * workload.data_parallel_size
    )
    backward = int(forward * backward_to_forward_ratio)
    return {"forward": forward, "backward": backward, "total": forward + backward}


def optimizer_update_flops(
    model: DecoderSpec, *, operations_per_parameter: int = 10
) -> int:
    """Estimate one dense optimizer update from an explicit operation budget."""

    _positive_integer(operations_per_parameter, "operations_per_parameter")
    return model.parameter_count * operations_per_parameter


def bytes_per_element(dtype: str) -> int:
    """Resolve the storage width of one supported tensor dtype."""

    if not isinstance(dtype, str):
        raise TypeError("dtype must be a string")
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as error:
        raise ResourceAccountingError(f"unsupported dtype: {dtype!r}") from error


def parameter_memory(model: DecoderSpec, *, dtype: str) -> dict[str, int]:
    """Return byte-exact storage for each declared parameter group."""

    width = bytes_per_element(dtype)
    groups = {name: count * width for name, count in model.parameter_groups().items()}
    groups["total"] = sum(groups.values())
    return groups


def training_state_memory(
    model: DecoderSpec,
    *,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    master_weights: bool,
) -> dict[str, int]:
    """Account for weights, gradients, Adam moments, and optional FP32 masters."""

    if not isinstance(master_weights, bool):
        raise TypeError("master_weights must be a boolean")
    count = model.parameter_count
    groups = {
        "parameters": count * bytes_per_element(parameter_dtype),
        "gradients": count * bytes_per_element(gradient_dtype),
        "optimizer_moments": count * bytes_per_element(optimizer_state_dtype) * 2,
        "master_parameters": count * 4 if master_weights else 0,
    }
    groups["total"] = sum(groups.values())
    return groups


def activation_memory(
    model: DecoderSpec, *, batch_size: int, sequence_length: int, dtype: str
) -> dict[str, int]:
    """Estimate saved dense layer activations from named tensor families."""

    _positive_integer(batch_size, "batch_size")
    _positive_integer(sequence_length, "sequence_length")
    width = bytes_per_element(dtype)
    tokens = batch_size * sequence_length
    layers = model.layer_count
    elements = {
        "residual_streams": tokens * layers * model.hidden_size * 2,
        "attention_projections": (
            tokens * layers * (model.hidden_size * 2 + model.kv_size * 2)
        ),
        "feed_forward_intermediates": (tokens * layers * model.feed_forward_size * 2),
    }
    groups = {name: count * width for name, count in elements.items()}
    groups["total"] = sum(groups.values())
    return groups


def checkpointed_activation_memory(
    model: DecoderSpec,
    *,
    batch_size: int,
    sequence_length: int,
    dtype: str,
    checkpoint_segments: int,
) -> dict[str, int]:
    """Estimate peak saved activations with uniform layer checkpointing.

    The estimate retains every segment boundary plus one segment's recomputed
    working set.  It intentionally excludes allocator and kernel workspaces.
    """

    _positive_integer(checkpoint_segments, "checkpoint_segments")
    if checkpoint_segments > model.layer_count:
        raise ResourceAccountingError("checkpoint_segments cannot exceed layer_count")
    if model.layer_count % checkpoint_segments:
        raise ResourceAccountingError(
            "layer_count must be divisible by checkpoint_segments"
        )
    width = bytes_per_element(dtype)
    tokens = _positive_integer(batch_size, "batch_size") * _positive_integer(
        sequence_length, "sequence_length"
    )
    full = activation_memory(
        model, batch_size=batch_size, sequence_length=sequence_length, dtype=dtype
    )["total"]
    layers_per_segment = model.layer_count // checkpoint_segments
    checkpoint_boundaries = (
        tokens * (checkpoint_segments + 1) * model.hidden_size * width
    )
    recompute_window = full * layers_per_segment // model.layer_count
    peak = checkpoint_boundaries + recompute_window
    return {
        "checkpoint_boundaries": checkpoint_boundaries,
        "recompute_window": recompute_window,
        "peak": peak,
        "saved_vs_uncheckpointed": full - peak,
    }
