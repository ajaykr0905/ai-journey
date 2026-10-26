import math

import pytest
import torch

from ai_journey.ddp_evaluation import Document, evaluate_documents
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel


def model_and_documents():
    model = GPT2LanguageModel(
        GPT2Config(
            vocab_size=256,
            context_length=4,
            embedding_dim=4,
            head_count=2,
            layer_count=1,
        )
    )
    documents = (Document("first", b"abcdef"), Document("second", b"XYZ"))
    return model, documents


def test_finite_logits_cannot_publish_infinite_likelihood(monkeypatch):
    model, documents = model_and_documents()
    model.train()
    model.h[0].eval()
    modes = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()

    def forward(inputs):
        limit = torch.finfo(torch.float32).max
        logits = torch.full((*inputs.shape, 256), -limit)
        logits[..., 0] = limit
        assert torch.isfinite(logits).all()
        return logits

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises(ValueError, match="token losses must be finite"):
        evaluate_documents(model, documents, batch_size=4)
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(saved, parameter.grad)
        for saved, parameter in zip(gradients, model.parameters(), strict=True)
    )


def test_valid_uniform_logits_preserve_target_weighting(monkeypatch):
    model, documents = model_and_documents()

    def forward(inputs):
        return torch.zeros((*inputs.shape, 256))

    monkeypatch.setattr(model, "forward", forward)
    result = evaluate_documents(model, documents, batch_size=4)
    assert result["tokens"] == 7
    assert result["mean_nll"] == pytest.approx(math.log(256))
    assert result["bits_per_byte"] == pytest.approx(8)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_token_loss_rejects_arithmetic_overflow_without_rejecting_stable_extremes(
    dtype,
):
    from ai_journey.ddp_metrics import token_nll

    limit = torch.finfo(dtype).max
    unstable = torch.tensor([[[limit, -limit]]], dtype=dtype, requires_grad=True)
    with pytest.raises(ValueError, match="token losses must be finite"):
        token_nll(unstable, torch.tensor([[1]]))
    assert unstable.grad is None
    stable = torch.full((1, 1, 2), limit, dtype=dtype, requires_grad=True)
    loss = token_nll(stable, torch.tensor([[1]]))
    assert float(loss.detach()[0, 0]) == pytest.approx(math.log(2), abs=0.01)
    loss.mean().backward()
    assert torch.isfinite(stable.grad).all()
