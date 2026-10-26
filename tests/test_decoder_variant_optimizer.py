import pytest
import torch

from ai_journey.decoder_variant_experiment import _build_matched_model
from ai_journey.decoder_variant_optimizer import (
    DecoderOptimizerConfig,
    build_decoder_optimizer,
    decoder_optimizer_groups,
)
from test_decoder_variant_integrity import config


def test_optimizer_grouping_is_complete_unique_and_excludes_norm_decay():
    model = _build_matched_model(config(), 7)
    groups = decoder_optimizer_groups(model)
    names = set(groups["decay_names"]) | set(groups["no_decay_names"])
    assert names == dict(model.named_parameters()).keys()
    assert set(groups["decay_names"]).isdisjoint(groups["no_decay_names"])
    assert groups["trainable_parameters"] == model.parameter_count
    assert "wte.weight" in groups["decay_names"]
    assert "lm_head.weight" not in names
    assert "ln_f.weight" in groups["no_decay_names"]
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = build_decoder_optimizer(
        model, learning_rate=0.01, config=DecoderOptimizerConfig(weight_decay=0.1)
    )
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        expected = before[name] * (0.999 if name in groups["decay_names"] else 1)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert model.wte.weight is model.lm_head.weight


@pytest.mark.parametrize(
    "controls",
    [
        {"beta1": 1.0},
        {"beta2": -0.1},
        {"weight_decay": True},
        {"epsilon": float("nan")},
        {"epsilon": 0},
    ],
)
def test_optimizer_rejects_invalid_controls(controls):
    with pytest.raises(ValueError, match="optimizer"):
        DecoderOptimizerConfig(**controls)
