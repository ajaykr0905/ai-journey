import pytest

from ai_journey.resource_accounting import DecoderSpec, ResourceAccountingError


def small_spec(**changes) -> DecoderSpec:
    values = {
        "vocab_size": 100,
        "layer_count": 2,
        "hidden_size": 16,
        "attention_heads": 4,
        "kv_heads": 2,
        "feed_forward_size": 32,
    }
    values.update(changes)
    return DecoderSpec(**values)


def test_decoder_spec_validates_head_geometry_and_types():
    assert small_spec().head_size == 4
    assert small_spec().kv_size == 8
    with pytest.raises(ResourceAccountingError, match="divisible"):
        small_spec(hidden_size=15)
    with pytest.raises(ResourceAccountingError, match="kv_heads"):
        small_spec(kv_heads=3)
    with pytest.raises(TypeError, match="integer"):
        small_spec(layer_count=True)


def test_parameter_inventory_is_complete_and_non_overlapping():
    spec = small_spec()
    assert spec.parameter_groups() == {
        "token_embedding": 1_600,
        "attention": 1_536,
        "feed_forward": 3_072,
        "block_norms": 64,
        "final_norm": 16,
        "output_projection": 1_600,
    }
    assert spec.parameter_count == 7_888


def test_tied_output_and_bias_parameters_are_explicit():
    tied = small_spec(tie_embeddings=True)
    biased = small_spec(use_bias=True)
    assert tied.parameter_count == small_spec().parameter_count - 1_600
    assert biased.parameter_count == small_spec().parameter_count + 256
