"""Expert decoder identity, masking, causal eval and cache parity."""

from dataclasses import replace

import pytest
import torch

from ai_journey.gpt2_training import GPT2TrainingSpec, language_model_loss
from ai_journey.moe_config import expected_moe_parameter_count
from ai_journey.moe_model import MoEDecoderModel
from ai_journey.training_models import (
    build_training_model,
    model_config_from_mapping,
    model_config_mapping,
    training_model_fingerprint,
)
from test_moe_config import small_config
from test_modern_training import modern_spec


@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("position", ["learned", "rope"])
def test_model_counts_contracts_and_explicit_per_forward_results(mlp, position):
    config = small_config(mlp_kind=mlp, position_kind=position, norm_kind="rmsnorm")
    model = build_training_model(config)
    assert isinstance(model, MoEDecoderModel)
    assert model.parameter_count == expected_moe_parameter_count(config)
    assert model.lm_head.weight is model.wte.weight
    assert model_config_from_mapping(model_config_mapping(config)) == config
    output = model.forward_training(
        torch.tensor([[1, 2, 3]]), token_mask=torch.tensor([[True, False, True]])
    )
    assert output.logits.shape == (1, 3, 256)
    assert output.routing[0]["valid_tokens"] == 2
    assert output.routing[0]["accepted_per_token"][0][1] == 0
    assert not hasattr(model, "last_stats")


def test_eval_is_causal_and_cache_matches_full_sequence_with_skewed_router():
    model = build_training_model(
        small_config(position_kind="rope", capacity_factor=0.01)
    ).eval()
    for block in model.h:
        if hasattr(block.mlp, "router"):
            block.mlp.router.projection.weight.data.zero_()
    tokens = torch.tensor([[1, 2, 3, 4]])
    full = model(tokens)
    torch.testing.assert_close(model(torch.tensor([[1, 2, 9, 9]]))[:, :2], full[:, :2])
    logits, cache = model.forward_with_cache(tokens[:, :2])
    later, _ = model.forward_with_cache(tokens[:, 2:], past_key_values=cache)
    torch.testing.assert_close(
        torch.cat((logits, later), 1), full, atol=1e-6, rtol=1e-5
    )
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model.forward_with_cache(tokens)


def test_fingerprint_binds_capacity_even_with_identical_tensor_state():
    config = small_config()
    left, right = (
        build_training_model(config),
        build_training_model(replace(config, capacity_factor=2)),
    )
    right.load_state_dict(left.state_dict())
    assert training_model_fingerprint(left) != training_model_fingerprint(right)


def test_dense_training_does_not_silently_omit_moe_auxiliary_objective():
    source, spec = modern_spec()
    config = small_config()
    with pytest.raises(ValueError, match="explicit auxiliary"):
        GPT2TrainingSpec(config, spec.training, spec.source)
    with pytest.raises(ValueError, match="explicit auxiliary"):
        language_model_loss(
            build_training_model(config),
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
        )


def test_moe_parameter_admission_rejects_before_model_constructor(monkeypatch):
    import ai_journey.training_models as models

    monkeypatch.setattr(
        models, "MoEDecoderModel", lambda _: pytest.fail("must not allocate")
    )
    with pytest.raises(ValueError, match="parameter budget"):
        models.build_training_model(small_config(), max_parameters=1)
