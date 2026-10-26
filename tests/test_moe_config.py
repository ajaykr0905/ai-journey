"""MoE architecture admission must be strict and analytically inspectable."""

from dataclasses import replace
import json

import pytest

from ai_journey.decoder_variant_resources import (
    expected_decoder_variant_parameter_count,
)
from ai_journey.gpt2_config import GPT2Config
from ai_journey.moe_config import (
    MoEConfig,
    expert_parameter_count,
    expected_moe_active_parameters,
    expected_moe_parameter_count,
    moe_config_from_mapping,
)


def small_config(**kwargs):
    return MoEConfig(GPT2Config(256, 8, 8, 2, 2), **kwargs)


@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
def test_json_contract_round_trip_and_parameter_arithmetic(mlp):
    config = small_config(mlp_kind=mlp, moe_layers=(0, 1))
    assert (
        moe_config_from_mapping(json.loads(json.dumps(config.to_mapping()))) == config
    )
    base = expected_decoder_variant_parameter_count(config)
    expert = (
        8 * 8**2 + 5 * 8 if mlp == "gelu" else 3 * 8 * config.resolved_swiglu_hidden_dim
    )
    assert expert_parameter_count(config) == expert
    assert expected_moe_parameter_count(config) == base + 2 * (3 * expert + 32)
    assert expected_moe_active_parameters(config) == base + 2 * (expert + 32)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expert_count", True),
        ("expert_count", 1),
        ("expert_count", 65),
        ("top_k", 1),
        ("top_k", True),
        ("capacity_factor", True),
        ("capacity_factor", 0),
        ("capacity_factor", float("nan")),
        ("capacity_factor", 17),
        ("min_capacity", 0),
        ("min_capacity", True),
        ("moe_layers", ()),
        ("moe_layers", (0, 0)),
        ("moe_layers", (1, 0)),
        ("moe_layers", (2,)),
        ("moe_layers", (True,)),
    ],
)
def test_invalid_controls_are_rejected(field, value):
    with pytest.raises(ValueError):
        small_config(**{field: value})


@pytest.mark.parametrize("field", ["expert_count", "capacity_factor", "moe_layers"])
def test_complete_control_identity_changes_even_with_same_tensor_shapes(field):
    config = small_config()
    changed = replace(
        config,
        **{field: {"expert_count": 5, "capacity_factor": 2, "moe_layers": (1,)}[field]},
    )
    assert config.fingerprint() != changed.fingerprint()


@pytest.mark.parametrize("mutation", ["unknown", "missing", "base"])
def test_parser_rejects_incomplete_or_unknown_fields(mutation):
    payload = small_config().to_mapping()
    if mutation == "unknown":
        payload["capacity_facter"] = 1
    elif mutation == "missing":
        del payload["top_k"]
    else:
        payload["gpt2"]["secret_extra"] = 1
    with pytest.raises(ValueError):
        moe_config_from_mapping(payload)


def test_huge_capacity_integer_is_a_validation_error_not_float_overflow():
    with pytest.raises(ValueError, match="capacity_factor"):
        small_config(capacity_factor=10**1000)


@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("positions", ["learned", "rope"])
def test_parameter_tensor_count_matches_actual_model_without_head_alias(
    mlp, norm, positions
):
    from ai_journey.decoder_variant_config import DecoderVariantConfig
    from ai_journey.decoder_variant_model import DecoderVariantModel
    from ai_journey.moe_model import MoEDecoderModel
    from ai_journey.moe_config import (
        expected_decoder_parameter_tensors,
        expected_moe_parameter_tensors,
    )

    config = small_config(mlp_kind=mlp, norm_kind=norm, position_kind=positions)
    assert expected_moe_parameter_tensors(config) == len(
        tuple(MoEDecoderModel(config).parameters())
    )
    dense = DecoderVariantConfig(
        config.gpt2, norm_kind=norm, mlp_kind=mlp, position_kind=positions
    )
    assert expected_decoder_parameter_tensors(dense) == len(
        tuple(DecoderVariantModel(dense).parameters())
    )
