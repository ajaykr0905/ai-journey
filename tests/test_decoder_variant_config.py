from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ai_journey.decoder_variant_config import (
    DecoderVariantConfig,
    DecoderVariantConfigError,
)
from ai_journey.gpt2_config import GPT2Config


def _gpt2_config(*, embedding_dim: int = 12, head_count: int = 3) -> GPT2Config:
    return GPT2Config(
        vocab_size=128,
        context_length=32,
        embedding_dim=embedding_dim,
        head_count=head_count,
        layer_count=2,
    )


def test_defaults_preserve_baseline_architecture_contract() -> None:
    baseline = _gpt2_config()
    baseline_fingerprint = baseline.fingerprint()

    variant = DecoderVariantConfig(gpt2=baseline)

    assert variant.gpt2 is baseline
    assert variant.norm_kind == "layernorm"
    assert variant.mlp_kind == "gelu"
    assert variant.position_kind == "learned"
    assert variant.rope_base == 10000.0
    assert variant.swiglu_hidden_dim is None
    assert variant.resolved_swiglu_hidden_dim is None
    assert baseline.fingerprint() == baseline_fingerprint


def test_rejects_a_non_gpt2_wrapped_config() -> None:
    with pytest.raises(TypeError, match="gpt2 must be GPT2Config"):
        DecoderVariantConfig(gpt2=object())  # type: ignore[arg-type]


def test_config_and_wrapped_gpt2_config_are_immutable() -> None:
    variant = DecoderVariantConfig(gpt2=_gpt2_config())

    with pytest.raises(FrozenInstanceError):
        variant.norm_kind = "rmsnorm"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        variant.gpt2.embedding_dim = 24  # type: ignore[misc]


@pytest.mark.parametrize(
    ("norm_kind", "mlp_kind", "position_kind"),
    [
        ("layernorm", "gelu", "learned"),
        ("rmsnorm", "gelu", "learned"),
        ("layernorm", "swiglu", "learned"),
        ("layernorm", "gelu", "rope"),
        ("rmsnorm", "swiglu", "rope"),
    ],
)
def test_accepts_independent_architecture_feature_combinations(
    norm_kind: str, mlp_kind: str, position_kind: str
) -> None:
    variant = DecoderVariantConfig(
        gpt2=_gpt2_config(),
        norm_kind=norm_kind,
        mlp_kind=mlp_kind,
        position_kind=position_kind,
    )

    assert (variant.norm_kind, variant.mlp_kind, variant.position_kind) == (
        norm_kind,
        mlp_kind,
        position_kind,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("norm_kind", "batchnorm"),
        ("mlp_kind", "relu"),
        ("position_kind", "alibi"),
        ("norm_kind", 1),
    ],
)
def test_rejects_unsupported_feature_kinds(field: str, value: object) -> None:
    with pytest.raises(DecoderVariantConfigError, match=field):
        DecoderVariantConfig(gpt2=_gpt2_config(), **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "rope_base", [1.0, 0.0, -2.0, float("inf"), float("nan"), True, 10**400]
)
def test_rope_base_must_be_finite_and_above_one(rope_base: object) -> None:
    with pytest.raises(DecoderVariantConfigError, match="rope_base"):
        DecoderVariantConfig(gpt2=_gpt2_config(), rope_base=rope_base)  # type: ignore[arg-type]


def test_rope_rejects_odd_head_width_without_restricting_learned_positions() -> None:
    baseline = _gpt2_config(embedding_dim=15, head_count=3)

    assert DecoderVariantConfig(gpt2=baseline).gpt2.head_dim == 5
    with pytest.raises(DecoderVariantConfigError, match="even head_dim"):
        DecoderVariantConfig(gpt2=baseline, position_kind="rope")


@pytest.mark.parametrize("hidden_dim", [0, -1, True, 2.5])
def test_swiglu_hidden_width_must_be_a_positive_integer(hidden_dim: object) -> None:
    with pytest.raises(DecoderVariantConfigError, match="swiglu_hidden_dim"):
        DecoderVariantConfig(
            gpt2=_gpt2_config(), mlp_kind="swiglu", swiglu_hidden_dim=hidden_dim
        )  # type: ignore[arg-type]


def test_swiglu_hidden_width_is_only_meaningful_for_swiglu() -> None:
    with pytest.raises(DecoderVariantConfigError, match="swiglu_hidden_dim"):
        DecoderVariantConfig(gpt2=_gpt2_config(), swiglu_hidden_dim=32)


def test_default_swiglu_width_matches_gelu_projection_parameter_budget() -> None:
    variant = DecoderVariantConfig(gpt2=_gpt2_config(), mlp_kind="swiglu")

    # GELU: C->4C->C uses 8*C^2 matrix parameters; SwiGLU: three CxH matrices.
    assert variant.resolved_swiglu_hidden_dim == 32
    assert 3 * 12 * variant.resolved_swiglu_hidden_dim == 8 * 12**2
    assert replace(variant, swiglu_hidden_dim=40).resolved_swiglu_hidden_dim == 40


def test_fingerprint_is_stable_and_changes_with_configuration() -> None:
    baseline = DecoderVariantConfig(gpt2=_gpt2_config())
    same = DecoderVariantConfig(gpt2=_gpt2_config())
    variants = (
        replace(baseline, norm_kind="rmsnorm"),
        replace(baseline, mlp_kind="swiglu"),
        replace(baseline, position_kind="rope"),
        replace(baseline, rope_base=20000.0),
        replace(baseline, gpt2=_gpt2_config(embedding_dim=16, head_count=4)),
    )

    assert baseline.fingerprint() == same.fingerprint()
    assert len(baseline.fingerprint()) == 64
    assert all(value in "0123456789abcdef" for value in baseline.fingerprint())
    assert (
        len({baseline.fingerprint(), *(variant.fingerprint() for variant in variants)})
        == 6
    )


def test_explicit_parity_width_has_same_effective_fingerprint() -> None:
    implicit = DecoderVariantConfig(gpt2=_gpt2_config(), mlp_kind="swiglu")
    explicit = replace(implicit, swiglu_hidden_dim=32)

    assert implicit.fingerprint() == explicit.fingerprint()


def test_integral_and_floating_rope_bases_have_one_fingerprint() -> None:
    integer = DecoderVariantConfig(gpt2=_gpt2_config(), rope_base=10000)
    floating = DecoderVariantConfig(gpt2=_gpt2_config(), rope_base=10000.0)

    assert integer.fingerprint() == floating.fingerprint()
