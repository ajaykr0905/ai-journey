"""Allocation-free parameter counts for opt-in decoder architectures."""

from __future__ import annotations

from .decoder_variant_config import DecoderVariantConfig
from .gpt2_model import expected_gpt2_parameter_count


def expected_decoder_variant_parameter_count(config: DecoderVariantConfig) -> int:
    """Count unique trainable values, excluding the tied output-head alias."""

    if not isinstance(config, DecoderVariantConfig):
        raise TypeError("config must be DecoderVariantConfig")

    base = config.gpt2
    width = base.embedding_dim
    layers = base.layer_count
    count = expected_gpt2_parameter_count(base)

    if config.norm_kind == "rmsnorm":
        # Two norms per block and one final norm each lose a width-sized bias.
        count -= (2 * layers + 1) * width
    if config.mlp_kind == "swiglu":
        hidden = config.resolved_swiglu_hidden_dim
        assert hidden is not None  # Guaranteed by the validated variant contract.
        # Three bias-free projections replace GPT-2's two biased MLP matrices.
        count += layers * (3 * width * hidden - (8 * width**2 + 5 * width))
    if config.position_kind == "rope":
        # RoPE uses no learned positional parameters, unlike the GPT-2 table.
        count -= base.context_length * width

    return count
