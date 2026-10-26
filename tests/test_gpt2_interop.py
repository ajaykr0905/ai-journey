from __future__ import annotations

from ai_journey.gpt2_config import GPT2_VARIANTS, GPT2Config
import pytest
import torch

from ai_journey.gpt2_interop import (
    GPT2InteropError,
    TensorSpec,
    convert_hugging_face_state_dict,
    expected_hugging_face_tensors,
    export_hugging_face_state,
    fingerprint_hugging_face_state,
    known_hugging_face_buffers,
    load_hugging_face_weights,
    normalize_hugging_face_checkpoint,
    validate_tied_embedding_alias,
    validate_hugging_face_state_dict,
)
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_model import DecoderVariantModel


def _tiny_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=11,
        context_length=4,
        embedding_dim=4,
        head_count=2,
        layer_count=2,
        embedding_dropout=0,
        attention_dropout=0,
        residual_dropout=0,
    )


def test_baseline_compatible_variant_import_and_export_preserve_exact_logits():
    baseline = GPT2LanguageModel(_tiny_config()).eval()
    variant = DecoderVariantModel(DecoderVariantConfig(_tiny_config())).eval()
    external = export_hugging_face_state(baseline)
    load_hugging_face_weights(variant, external)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(baseline(tokens), variant(tokens))
    exported = export_hugging_face_state(variant)
    assert external.keys() == exported.keys()
    assert all(torch.equal(value, exported[name]) for name, value in external.items())
    assert variant.lm_head.weight is variant.wte.weight


@pytest.mark.parametrize(
    "choices",
    [{"norm_kind": "rmsnorm"}, {"mlp_kind": "swiglu"}, {"position_kind": "rope"}],
)
def test_incompatible_variant_rejects_gpt2_weights_before_mutation(choices):
    model = DecoderVariantModel(DecoderVariantConfig(_tiny_config(), **choices))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(GPT2InteropError, match="baseline-compatible"):
        load_hugging_face_weights(model, _valid_state_dict())
    with pytest.raises(GPT2InteropError, match="baseline-compatible"):
        export_hugging_face_state(model)
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_expected_tensor_plan_covers_every_parameter_and_tied_head_alias() -> None:
    specs = expected_hugging_face_tensors(_tiny_config())

    assert len(specs) == 29
    assert specs["transformer.wte.weight"] == TensorSpec((11, 4))
    assert specs["transformer.wpe.weight"] == TensorSpec((4, 4))
    assert specs["lm_head.weight"] == TensorSpec((11, 4))
    assert specs["transformer.h.1.ln_2.bias"] == TensorSpec((4,))


def test_conv1d_weights_are_the_only_transposed_tensor_classes() -> None:
    specs = expected_hugging_face_tensors(_tiny_config())

    transposed = {key for key, spec in specs.items() if spec.transpose}

    assert transposed == {
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.1.attn.c_proj.weight",
        "transformer.h.1.mlp.c_fc.weight",
        "transformer.h.1.mlp.c_proj.weight",
    }


def test_gpt2_124m_plan_has_the_expected_tensor_count() -> None:
    specs = expected_hugging_face_tensors(GPT2_VARIANTS["gpt2"])

    assert len(specs) == 149
    assert specs["transformer.h.11.attn.c_attn.weight"].shape == (768, 2_304)
    assert specs["transformer.h.11.mlp.c_proj.weight"].shape == (3_072, 768)


def _valid_state_dict() -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros(spec.shape, dtype=torch.float32)
        for key, spec in expected_hugging_face_tensors(_tiny_config()).items()
    }


def test_validates_complete_finite_hugging_face_state() -> None:
    state = _valid_state_dict()
    state["transformer.h.0.attn.bias"] = torch.ones(1, 1, 4, 4)

    ignored = validate_hugging_face_state_dict(state, _tiny_config())

    assert ignored == ("transformer.h.0.attn.bias",)
    assert len(known_hugging_face_buffers(_tiny_config())) == 4


def test_normalizes_public_legacy_checkpoint_names_and_synthesizes_tied_head() -> None:
    external = _valid_state_dict()
    legacy = {
        key.removeprefix("transformer."): value
        for key, value in external.items()
        if key != "lm_head.weight"
    }
    legacy["h.0.attn.bias"] = torch.ones(1, 1, 4, 4)

    normalized = normalize_hugging_face_checkpoint(legacy, _tiny_config())

    assert set(expected_hugging_face_tensors(_tiny_config())) <= set(normalized)
    assert normalized["lm_head.weight"] is normalized["transformer.wte.weight"]
    assert "transformer.h.0.attn.bias" in normalized


def test_rejects_mixed_or_unexpected_legacy_checkpoint_names() -> None:
    external = _valid_state_dict()
    legacy = {
        key.removeprefix("transformer."): value
        for key, value in external.items()
        if key != "lm_head.weight"
    }
    mixed = {**legacy, "transformer.wte.weight": legacy["wte.weight"]}
    unexpected = {**legacy, "secret.weight": torch.ones(1)}

    with pytest.raises(GPT2InteropError, match="must not mix"):
        normalize_hugging_face_checkpoint(mixed, _tiny_config())
    with pytest.raises(GPT2InteropError, match="unexpected legacy"):
        normalize_hugging_face_checkpoint(unexpected, _tiny_config())


def test_rejects_missing_and_unexpected_state_dict_keys() -> None:
    missing = _valid_state_dict()
    del missing["transformer.ln_f.bias"]
    unexpected = _valid_state_dict()
    unexpected["transformer.secret.weight"] = torch.zeros(1)

    with pytest.raises(GPT2InteropError, match="missing.*ln_f.bias"):
        validate_hugging_face_state_dict(missing, _tiny_config())
    with pytest.raises(GPT2InteropError, match="unexpected.*secret"):
        validate_hugging_face_state_dict(unexpected, _tiny_config())


def test_rejects_wrong_shape_dtype_and_nonfinite_tensors() -> None:
    wrong_shape = _valid_state_dict()
    wrong_shape["transformer.wpe.weight"] = torch.zeros(3, 4)
    integer = _valid_state_dict()
    integer["transformer.ln_f.weight"] = torch.ones(4, dtype=torch.int64)
    nonfinite = _valid_state_dict()
    nonfinite["transformer.ln_f.bias"][2] = float("nan")

    with pytest.raises(GPT2InteropError, match="shape"):
        validate_hugging_face_state_dict(wrong_shape, _tiny_config())
    with pytest.raises(GPT2InteropError, match="floating-point"):
        validate_hugging_face_state_dict(integer, _tiny_config())
    with pytest.raises(GPT2InteropError, match="finite"):
        validate_hugging_face_state_dict(nonfinite, _tiny_config())


def test_requires_the_language_head_to_match_the_token_embedding_exactly() -> None:
    state = _valid_state_dict()
    state["lm_head.weight"][0, 0] = 1

    with pytest.raises(GPT2InteropError, match="exactly equal"):
        validate_hugging_face_state_dict(state, _tiny_config())


def test_tied_embedding_check_rejects_missing_alias_inputs() -> None:
    with pytest.raises(GPT2InteropError, match="require wte and lm_head"):
        validate_tied_embedding_alias({})


def test_converts_names_and_transposes_only_hugging_face_conv1d_weights() -> None:
    state = _valid_state_dict()
    state["transformer.h.0.attn.c_attn.weight"] = torch.arange(48).view(4, 12).float()
    state["transformer.h.0.attn.c_proj.weight"] = torch.arange(16).view(4, 4).float()

    converted = convert_hugging_face_state_dict(state, _tiny_config())

    assert "transformer.wte.weight" not in converted
    assert converted["wte.weight"].shape == (11, 4)
    assert converted["h.0.attn.c_attn.weight"].shape == (12, 4)
    torch.testing.assert_close(
        converted["h.0.attn.c_attn.weight"],
        state["transformer.h.0.attn.c_attn.weight"].T,
    )
    torch.testing.assert_close(
        converted["h.0.attn.c_proj.weight"],
        state["transformer.h.0.attn.c_proj.weight"].T,
    )


def test_conversion_does_not_share_writable_tensor_storage() -> None:
    state = _valid_state_dict()
    converted = convert_hugging_face_state_dict(state, _tiny_config())

    converted["ln_f.weight"][0] = 9

    assert state["transformer.ln_f.weight"][0] == 0


def test_loads_complete_hugging_face_state_and_preserves_weight_tying() -> None:
    model = GPT2LanguageModel(_tiny_config())
    state = _valid_state_dict()
    tied = torch.arange(44, dtype=torch.float32).view(11, 4) / 100
    state["transformer.wte.weight"] = tied.clone()
    state["lm_head.weight"] = tied.clone()
    state["transformer.wpe.weight"] = torch.arange(16).view(4, 4).float()

    load_hugging_face_weights(model, state)

    torch.testing.assert_close(model.wte.weight, tied)
    torch.testing.assert_close(model.wpe.weight, state["transformer.wpe.weight"])
    assert model.lm_head.weight is model.wte.weight


def test_invalid_external_state_does_not_partially_mutate_the_model() -> None:
    torch.manual_seed(45)
    model = GPT2LanguageModel(_tiny_config())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = _valid_state_dict()
    state["transformer.h.1.mlp.c_fc.weight"] = torch.zeros(3, 3)

    with pytest.raises(GPT2InteropError, match="shape"):
        load_hugging_face_weights(model, state)

    for key, expected in before.items():
        assert torch.equal(model.state_dict()[key], expected), key


def test_loading_rejects_implicit_dtype_conversion() -> None:
    model = GPT2LanguageModel(_tiny_config())
    state = {key: value.double() for key, value in _valid_state_dict().items()}

    with pytest.raises(GPT2InteropError, match="dtype"):
        load_hugging_face_weights(model, state)


def test_state_fingerprint_is_order_independent_and_covers_tensor_values() -> None:
    state = _valid_state_dict()
    reversed_state = dict(reversed(list(state.items())))

    baseline = fingerprint_hugging_face_state(state, _tiny_config())
    reordered = fingerprint_hugging_face_state(reversed_state, _tiny_config())
    state["transformer.h.1.mlp.c_proj.bias"][2] = 0.25
    changed = fingerprint_hugging_face_state(state, _tiny_config())

    assert baseline == reordered
    assert baseline != changed
    assert len(baseline) == 64


def test_state_fingerprint_excludes_nonlearned_legacy_masks() -> None:
    state = _valid_state_dict()
    baseline = fingerprint_hugging_face_state(state, _tiny_config())
    state["transformer.h.0.attn.bias"] = torch.zeros(1, 1, 4, 4)

    assert fingerprint_hugging_face_state(state, _tiny_config()) == baseline


def test_import_then_export_round_trips_every_hugging_face_tensor() -> None:
    state = _valid_state_dict()
    generator = torch.Generator().manual_seed(45)
    for key, tensor in state.items():
        tensor.copy_(torch.randn(tensor.shape, generator=generator))
    state["lm_head.weight"] = state["transformer.wte.weight"].clone()
    model = GPT2LanguageModel(_tiny_config())

    load_hugging_face_weights(model, state)
    exported = export_hugging_face_state(model)

    assert set(exported) == set(state)
    for key, expected in state.items():
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)


def test_exported_state_has_independent_writable_storage() -> None:
    model = GPT2LanguageModel(_tiny_config())
    exported = export_hugging_face_state(model)

    exported["transformer.wte.weight"][0, 0] += 10

    assert not torch.equal(exported["transformer.wte.weight"], model.wte.weight)
