"""Tagged grouped-query models compose without changing old architecture bytes."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gqa_config import (
    GQAConfig,
    cache_head_count,
    expected_gqa_parameter_count,
    gqa_config_from_mapping,
)
from ai_journey.moe_config import MoEConfig
from ai_journey.training_models import (
    base_model_config,
    build_training_model,
    model_config_from_mapping,
    model_config_mapping,
    training_model_fingerprint,
)
from test_grouped_rotary_attention import config


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_tagged_dense_and_expert_model_roundtrips_counts_and_real_forward(
    sparse, kv_heads
):
    kind = MoEConfig if sparse else DecoderVariantConfig
    base = kind(config(), position_kind="rope", norm_kind="rmsnorm", mlp_kind="swiglu")
    grouped = GQAConfig(base, kv_heads)
    assert (
        gqa_config_from_mapping(json.loads(json.dumps(grouped.to_mapping()))) == grouped
    )
    assert model_config_from_mapping(model_config_mapping(grouped)) == grouped
    model = build_training_model(grouped)
    assert model.parameter_count == expected_gqa_parameter_count(grouped)
    assert base_model_config(model) == config()
    assert cache_head_count(model) == kv_heads
    assert model.lm_head.weight is model.wte.weight
    output = model.forward_training(
        torch.tensor([[1, 2, 3]]), token_mask=torch.tensor([[True, False, True]])
    )
    assert output.logits.shape == (1, 3, 256)
    if sparse:
        assert output.routing[0]["valid_tokens"] == 2
        assert all(
            name.startswith("decoder.h.") for name in output.idle_expert_parameters
        )
    else:
        assert output.routing == () and output.balance_loss == 0


def test_legacy_dense_mapping_and_fingerprint_remain_exactly_untouched():
    base = DecoderVariantConfig(config(), position_kind="rope")
    before = json.dumps(model_config_mapping(base), sort_keys=True)
    model = build_training_model(base)
    fingerprint = training_model_fingerprint(model)
    build_training_model(GQAConfig(base, 2))
    assert json.dumps(model_config_mapping(base), sort_keys=True) == before
    assert model_config_mapping(base) == {
        "kind": "decoder_variant",
        "config": asdict(base),
    }
    assert training_model_fingerprint(model) == fingerprint
    assert cache_head_count(base) == config().head_count


def test_gqa_fingerprint_binds_rotary_base_with_shape_compatible_state():
    base = DecoderVariantConfig(config(), position_kind="rope")
    first = build_training_model(GQAConfig(base, 2))
    second = build_training_model(GQAConfig(replace(base, rope_base=1000), 2))
    second.load_state_dict(first.state_dict())
    assert training_model_fingerprint(first) != training_model_fingerprint(second)


@pytest.mark.parametrize("heads", [0, True, 3, 5])
def test_invalid_kv_heads_rejected(heads):
    with pytest.raises(ValueError):
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), heads)


def test_learned_position_and_nested_grouped_wrappers_are_not_silent_interventions():
    with pytest.raises(ValueError, match="rotary"):
        GQAConfig(DecoderVariantConfig(config()), 2)
    with pytest.raises(TypeError):
        GQAConfig(GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 2), 1)


def test_unknown_grouped_config_fields_are_rejected():
    mapping = GQAConfig(
        DecoderVariantConfig(config(), position_kind="rope"), 2
    ).to_mapping()
    mapping["kv_headz"] = 2
    with pytest.raises(ValueError):
        gqa_config_from_mapping(mapping)
