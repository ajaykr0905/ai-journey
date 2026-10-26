"""Finite token losses do not imply representable additive/derived metrics."""

import math

import pytest
import torch

from ai_journey.ddp_evaluation import Document, evaluate_documents
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def measured_model(monkeypatch, *, context, loss):
    model = GPT2LanguageModel(GPT2Config(256, context, 4, 2, 1)).double().train()
    model.h[0].eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    def forward(inputs):
        logits = torch.full((*inputs.shape, 256), -loss, dtype=torch.float64)
        logits[..., 0] = 0
        assert torch.isfinite(logits).all()
        return logits

    monkeypatch.setattr(model, "forward", forward)
    return model


@pytest.mark.parametrize(
    "context,content,loss",
    [
        (4, b"\x01\x01\x01\x01", 8e307),  # one tensor reduction
        (1, b"\x01\x01\x01\x01", 8e307),  # finite per-batch totals overflow
        (4, b"\x01\x01", 1.5e308),  # finite sum/mean but bits-per-byte overflows
    ],
)
def test_unrepresentable_evaluation_totals_reject_and_preserve_caller(
    monkeypatch, context, content, loss
):
    model = measured_model(monkeypatch, context=context, loss=loss)
    modes = [module.training for module in model.modules()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="evaluation totals must be finite"):
        evaluate_documents(model, [Document("public", content)], batch_size=1)
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(before, parameter.grad)
        for before, parameter in zip(gradients, model.parameters(), strict=True)
    )


@pytest.mark.parametrize("loss", [2.0, 8e307])
def test_representable_extreme_metrics_remain_valid(monkeypatch, loss):
    model = measured_model(monkeypatch, context=4, loss=loss)
    report = evaluate_documents(model, [Document("public", b"\x01\x01")])
    expected = loss + math.log1p(255 * math.exp(-loss))
    assert report["tokens"] == 1
    assert report["loss_sum"] == pytest.approx(expected)
    assert report["mean_nll"] == pytest.approx(expected)
    assert report["bits_per_byte"] == pytest.approx(expected / math.log(2))
    assert all(
        math.isfinite(report[key]) for key in ("loss_sum", "mean_nll", "bits_per_byte")
    )
