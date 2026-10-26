"""Training architecture contracts must remain strict and allocation bounded."""

from dataclasses import asdict, replace
from itertools import product
import math

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config, GPT2_VARIANTS
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import gpt2_model_fingerprint
from ai_journey.training_models import (
    base_model_config,
    build_training_model,
    expected_training_parameters,
    model_config_from_mapping,
    model_config_mapping,
    training_model_fingerprint,
    MAX_CPU_MODEL_LAYERS,
    MAX_CPU_ATTENTION_MASK_BYTES,
)


@pytest.mark.parametrize(
    "norm,mlp,positions",
    product(("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")),
)
def test_every_modern_contract_round_trips_builds_and_counts(norm, mlp, positions):
    base = GPT2Config(256, 4, 8, 2, 1)
    config = DecoderVariantConfig(base, norm, mlp, positions)
    parsed = model_config_from_mapping(model_config_mapping(config))
    model = build_training_model(parsed)
    assert parsed == config
    assert base_model_config(model) == base
    assert model.parameter_count == expected_training_parameters(config)
    assert model.lm_head.weight is model.wte.weight


def test_baseline_contract_and_fingerprint_remain_compatible():
    config = GPT2Config(256, 4, 8, 2, 1)
    model = GPT2LanguageModel(config)
    assert model_config_mapping(config) == asdict(config)
    assert model_config_from_mapping(asdict(config)) == config
    assert training_model_fingerprint(model) == gpt2_model_fingerprint(model)


def test_fingerprint_distinguishes_same_tensor_state_with_different_rotary_base():
    config = DecoderVariantConfig(GPT2Config(256, 4, 8, 2, 1), position_kind="rope")
    left = build_training_model(config)
    right = build_training_model(replace(config, rope_base=1000))
    right.load_state_dict(left.state_dict())
    assert training_model_fingerprint(left) != training_model_fingerprint(right)


def test_parameter_budget_rejects_before_constructor(monkeypatch):
    config = DecoderVariantConfig(GPT2Config(256, 4, 8, 2, 1))

    def unexpected(*args, **kwargs):
        raise AssertionError("allocation must not occur")

    monkeypatch.setattr("ai_journey.training_models.DecoderVariantModel", unexpected)
    with pytest.raises(ValueError, match="parameter budget"):
        build_training_model(config, max_parameters=1)


@pytest.mark.parametrize("mutate", ["unknown", "missing", "kind", "base"])
def test_model_parser_rejects_unknown_incomplete_or_misspelled_contracts(mutate):
    payload = model_config_mapping(DecoderVariantConfig(GPT2Config(256, 4, 8, 2, 1)))
    if mutate == "unknown":
        payload["config"]["dropuot"] = 0
    elif mutate == "missing":
        del payload["config"]["norm_kind"]
    elif mutate == "kind":
        payload["kind"] = "llama"
    else:
        payload["config"]["gpt2"]["extra"] = True
    with pytest.raises(ValueError):
        model_config_from_mapping(payload)


@pytest.mark.parametrize("modern", [False, True])
def test_tiny_parameter_model_cannot_allocate_unbounded_attention_mask(
    monkeypatch, modern
):
    base = GPT2Config(256, 500000, 2 if modern else 1, 1, 1)
    config = (
        DecoderVariantConfig(base, norm_kind="rmsnorm", position_kind="rope")
        if modern
        else base
    )
    assert expected_training_parameters(config) < 5_000_000

    def unexpected(*args, **kwargs):
        raise AssertionError("no torch.ones allocation may occur")

    monkeypatch.setattr(torch, "ones", unexpected)
    with pytest.raises(ValueError, match="attention-mask construction"):
        build_training_model(config)


@pytest.mark.parametrize("modern", [False, True])
def test_attention_construction_budget_boundary_requires_no_large_allocation(
    monkeypatch, modern
):
    import ai_journey.training_models as models

    context = math.isqrt(MAX_CPU_ATTENTION_MASK_BYTES // 2)
    base = GPT2Config(256, context, 2 if modern else 1, 1, 1)
    config = DecoderVariantConfig(base, position_kind="rope") if modern else base
    constructed = []
    sentinel = object()

    def constructor(contract):
        constructed.append(contract)
        return sentinel

    monkeypatch.setattr(
        models, "DecoderVariantModel" if modern else "GPT2LanguageModel", constructor
    )
    assert build_training_model(config) is sentinel
    excessive_base = replace(base, context_length=context + 1)
    excessive = replace(config, gpt2=excessive_base) if modern else excessive_base
    with pytest.raises(ValueError, match="attention-mask construction"):
        build_training_model(excessive)
    assert constructed == [config]


def test_decoder_block_work_budget_is_independent_of_parameter_count(monkeypatch):
    import ai_journey.training_models as models

    sentinel = object()
    monkeypatch.setattr(models, "GPT2LanguageModel", lambda contract: sentinel)
    boundary = GPT2Config(256, 1, 1, 1, MAX_CPU_MODEL_LAYERS)
    assert build_training_model(boundary) is sentinel
    excessive = replace(boundary, layer_count=MAX_CPU_MODEL_LAYERS + 1)
    assert expected_training_parameters(excessive) < 5_000_000
    with pytest.raises(ValueError, match="layer work budget"):
        build_training_model(excessive)


@pytest.mark.parametrize("config", GPT2_VARIANTS.values())
def test_standard_gpt2_architectures_retain_admission_with_explicit_parameter_budget(
    monkeypatch, config
):
    import ai_journey.training_models as models

    sentinel = object()
    monkeypatch.setattr(models, "GPT2LanguageModel", lambda contract: sentinel)
    assert (
        build_training_model(
            config, max_parameters=expected_training_parameters(config)
        )
        is sentinel
    )
