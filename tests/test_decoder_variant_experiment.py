from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_experiment import (
    _build_matched_model,
    run_variant_comparison,
)
from ai_journey.gpt2_config import GPT2Config


def _config() -> DecoderVariantConfig:
    return DecoderVariantConfig(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
    )


def _run(**overrides: object):
    params = {
        "train_bytes": b"architecture comparisons need identical training windows" * 2,
        "heldout_bytes": b"separate heldout bytes for measured loss" * 2,
        "variants": {
            "baseline": _config(),
            "rmsnorm": replace(_config(), norm_kind="rmsnorm"),
            "swiglu": replace(_config(), mlp_kind="swiglu"),
            "rope": replace(_config(), position_kind="rope"),
        },
        "seed": 52,
        "steps": 2,
        "batch_size": 2,
        "learning_rate": 0.001,
        "evaluation_windows": 3,
    }
    params.update(overrides)
    return run_variant_comparison(**params)


def test_comparison_is_repeatable_and_uses_shared_evidence() -> None:
    before = torch.get_rng_state().clone()

    first = _run()
    second = _run()

    assert first == second
    assert torch.equal(torch.get_rng_state(), before)
    assert [result.name for result in first.variants] == [
        "baseline",
        "rmsnorm",
        "swiglu",
        "rope",
    ]
    assert first.heldout_windows == 3
    assert len(first.train_sha256) == len(first.batch_offsets_sha256) == 64
    assert all(result.parameter_count > 0 for result in first.variants)
    assert all(result.final_train_nll > 0 for result in first.variants)
    assert all(result.heldout_nll > 0 for result in first.variants)


def test_shared_parameters_start_identically_across_all_architecture_arms() -> None:
    base = _config()
    base = replace(base, gpt2=replace(base.gpt2, layer_count=2))
    before = torch.get_rng_state().clone()
    models = {}
    for norm, mlp, position in product(
        ("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")
    ):
        config = replace(base, norm_kind=norm, mlp_kind=mlp, position_kind=position)
        models[(norm, mlp, position)] = _build_matched_model(config, seed=52)

    assert torch.equal(torch.get_rng_state(), before)
    reference_by_name = {}
    for model in models.values():
        assert model.wte.weight is model.lm_head.weight
        for name, parameter in model.named_parameters():
            if name in reference_by_name:
                assert torch.equal(parameter, reference_by_name[name]), name
            else:
                reference_by_name[name] = parameter

    baseline = models[("layernorm", "gelu", "learned")]
    rope = models[("layernorm", "gelu", "rope")]
    swiglu = models[("layernorm", "swiglu", "learned")]
    assert torch.equal(baseline.h[0].attn.c_attn.weight, rope.h[0].attn.c_attn.weight)
    assert torch.equal(baseline.h[1].attn.c_attn.weight, swiglu.h[1].attn.c_attn.weight)


def test_training_offsets_do_not_depend_on_variant_order() -> None:
    first = _run()
    reversed_variants = dict(
        reversed(
            list(
                {
                    "baseline": _config(),
                    "rmsnorm": replace(_config(), norm_kind="rmsnorm"),
                    "swiglu": replace(_config(), mlp_kind="swiglu"),
                    "rope": replace(_config(), position_kind="rope"),
                }.items()
            )
        )
    )

    second = _run(variants=reversed_variants)

    assert first.batch_offsets_sha256 == second.batch_offsets_sha256
    assert {arm.name: arm for arm in first.variants} == {
        arm.name: arm for arm in second.variants
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_bytes": b"abc"}, "context_length"),
        ({"heldout_bytes": b"abc"}, "context_length"),
        ({"steps": 0}, "steps"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": -1}, "seed"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"evaluation_windows": 0}, "evaluation_windows"),
        ({"variants": {}}, "variants"),
        (
            {
                "heldout_bytes": b"architecture comparisons need identical training windows"
                * 2
            },
            "differ",
        ),
        ({"variants": {"a": _config(), "b": _config()}}, "distinct"),
    ],
)
def test_comparison_rejects_invalid_protocol(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _run(**overrides)


def test_comparison_rejects_mismatched_base_and_dropout() -> None:
    base = _config()
    changed = replace(base, gpt2=replace(base.gpt2, embedding_dim=12, head_count=3))
    with pytest.raises(ValueError, match="share one"):
        _run(variants={"a": base, "b": changed})
    with pytest.raises(ValueError, match="dropout"):
        _run(
            variants={"a": replace(base, gpt2=replace(base.gpt2, residual_dropout=0.1))}
        )
