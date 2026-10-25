from __future__ import annotations

from itertools import product

import pytest

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_model import DecoderVariantModel
from ai_journey.decoder_variant_resources import (
    expected_decoder_variant_parameter_count,
)
from ai_journey.gpt2_config import GPT2_VARIANTS, GPT2Config
from ai_journey.gpt2_model import expected_gpt2_parameter_count


def _tiny_gpt2_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=17,
        context_length=6,
        embedding_dim=8,
        head_count=2,
        layer_count=2,
    )


@pytest.mark.parametrize(
    ("norm_kind", "mlp_kind", "position_kind"),
    list(product(("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope"))),
)
def test_analytical_count_matches_unique_model_parameters_for_all_variants(
    norm_kind: str, mlp_kind: str, position_kind: str
) -> None:
    config = DecoderVariantConfig(
        gpt2=_tiny_gpt2_config(),
        norm_kind=norm_kind,  # type: ignore[arg-type]
        mlp_kind=mlp_kind,  # type: ignore[arg-type]
        position_kind=position_kind,  # type: ignore[arg-type]
        swiglu_hidden_dim=13 if mlp_kind == "swiglu" else None,
    )

    assert expected_decoder_variant_parameter_count(config) == (
        DecoderVariantModel(config).parameter_count
    )


def test_each_architecture_delta_is_exact_and_additive() -> None:
    base = _tiny_gpt2_config()
    baseline = expected_gpt2_parameter_count(base)
    width = base.embedding_dim
    layers = base.layer_count
    hidden = 13

    assert expected_decoder_variant_parameter_count(DecoderVariantConfig(base)) == (
        baseline
    )
    assert (
        expected_decoder_variant_parameter_count(
            DecoderVariantConfig(base, norm_kind="rmsnorm")
        )
        == baseline - (2 * layers + 1) * width
    )
    assert expected_decoder_variant_parameter_count(
        DecoderVariantConfig(base, mlp_kind="swiglu", swiglu_hidden_dim=hidden)
    ) == baseline + layers * (3 * width * hidden - (8 * width**2 + 5 * width))
    assert (
        expected_decoder_variant_parameter_count(
            DecoderVariantConfig(base, position_kind="rope")
        )
        == baseline - base.context_length * width
    )
    assert expected_decoder_variant_parameter_count(
        DecoderVariantConfig(
            base,
            norm_kind="rmsnorm",
            mlp_kind="swiglu",
            position_kind="rope",
            swiglu_hidden_dim=hidden,
        )
    ) == (
        baseline
        - (2 * layers + 1) * width
        + layers * (3 * width * hidden - (8 * width**2 + 5 * width))
        - base.context_length * width
    )


def test_default_swiglu_width_matches_instantiated_model() -> None:
    config = DecoderVariantConfig(gpt2=_tiny_gpt2_config(), mlp_kind="swiglu")

    assert config.resolved_swiglu_hidden_dim == 21
    assert expected_decoder_variant_parameter_count(config) == (
        DecoderVariantModel(config).parameter_count
    )


def test_smallest_even_head_with_single_hidden_channel_matches_model() -> None:
    base = GPT2Config(
        vocab_size=3,
        context_length=2,
        embedding_dim=2,
        head_count=1,
        layer_count=1,
    )
    config = DecoderVariantConfig(
        base,
        norm_kind="rmsnorm",
        mlp_kind="swiglu",
        position_kind="rope",
        swiglu_hidden_dim=1,
    )

    assert expected_decoder_variant_parameter_count(config) == (
        DecoderVariantModel(config).parameter_count
    )


def test_published_gpt2_count_remains_124_million_without_model_allocation() -> None:
    base = GPT2_VARIANTS["gpt2"]

    assert expected_decoder_variant_parameter_count(DecoderVariantConfig(base)) == (
        124_439_808
    )


def test_expected_count_requires_decoder_variant_config() -> None:
    with pytest.raises(TypeError, match="DecoderVariantConfig"):
        expected_decoder_variant_parameter_count(_tiny_gpt2_config())  # type: ignore[arg-type]
