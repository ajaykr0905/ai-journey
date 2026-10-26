import copy

import pytest
import torch

from ai_journey.decoder_variant_experiment import _build_matched_model
from ai_journey.decoder_variant_optimizer import (
    DecoderOptimizerConfig,
    build_decoder_optimizer,
)
from ai_journey.decoder_variant_training import train_decoder_step
from ai_journey.transformer_lab import model_fingerprint
from test_decoder_variant_integrity import config


def setup():
    model = _build_matched_model(config(), 7)
    optimizer = build_decoder_optimizer(
        model, learning_rate=0.001, config=DecoderOptimizerConfig()
    )
    inputs = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    return model, optimizer, inputs, inputs + 1


def test_training_records_clipping_and_clears_gradients():
    model, optimizer, inputs, targets = setup()
    before = model_fingerprint(model)
    metric = train_decoder_step(
        model, optimizer, inputs, targets, max_gradient_norm=0.01
    )
    assert metric["target_tokens"] == 8
    assert metric["clipped"]
    assert metric["gradient_norm_before_clip"] > metric["gradient_norm_after_clip"]
    assert metric["gradient_norm_after_clip"] <= 0.0100001
    assert model_fingerprint(model) != before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonfinite_gradients_fail_before_optimizer_update():
    model, optimizer, inputs, targets = setup()
    before = model_fingerprint(model)
    handle = model.wte.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(FloatingPointError, match="gradients"):
        train_decoder_step(model, optimizer, inputs, targets)
    handle.remove()
    assert model_fingerprint(model) == before
    assert not optimizer.state


def test_failed_update_rolls_back_model_and_optimizer(monkeypatch):
    model, optimizer, inputs, targets = setup()
    before = model_fingerprint(model)
    saved = copy.deepcopy(optimizer.state_dict())
    step = optimizer.step

    def corrupt():
        step()
        with torch.no_grad():
            model.wte.weight[0, 0] = float("inf")

    monkeypatch.setattr(optimizer, "step", corrupt)
    with pytest.raises(FloatingPointError, match="state"):
        train_decoder_step(model, optimizer, inputs, targets)
    assert model_fingerprint(model) == before
    assert optimizer.state_dict() == saved
