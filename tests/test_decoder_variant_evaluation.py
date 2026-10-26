import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ai_journey.ddp_evaluation import Document
from ai_journey.decoder_variant_evaluation import evaluate_decoder_documents


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(gpt2=SimpleNamespace(context_length=4))
        self.bias = nn.Parameter(torch.arange(256, dtype=torch.float32) / 100)
        self.child = nn.Dropout(0)
        self.batches = []

    def forward(self, inputs):
        self.batches.append(inputs.clone())
        return self.bias.expand(*inputs.shape, 256)


def test_full_evaluation_covers_tail_and_each_document_target_once():
    model = Probe()
    documents = (Document("a", b"aaaaaaaaaa"), Document("b", b"bbbbb"))
    first = evaluate_decoder_documents(model, documents, batch_size=1)
    second = evaluate_decoder_documents(model, documents, batch_size=3)
    assert first == second
    assert first["target_tokens"] == 13
    assert first["mean_nll"] == first["nll_sum"] / 13
    assert first["bits_per_byte"] == first["mean_nll"] / math.log(2)
    for batch in model.batches:
        for row in batch.tolist():
            assert len(set(row) - {0}) == 1
    changed = evaluate_decoder_documents(
        model, (Document("a", b"aaaaaaaaaz"), documents[1])
    )
    assert changed["nll_sum"] != first["nll_sum"]


def test_evaluation_preserves_modes_gradients_rng_and_parameters():
    model = Probe().train()
    model.child.eval()
    model.bias.grad = torch.ones_like(model.bias)
    parameters = model.bias.detach().clone()
    gradient = model.bias.grad.clone()
    rng = torch.get_rng_state().clone()
    evaluate_decoder_documents(model, (Document("a", b"test material"),))
    assert model.training and not model.child.training
    assert torch.equal(model.bias, parameters)
    assert torch.equal(model.bias.grad, gradient)
    assert torch.equal(torch.get_rng_state(), rng)


def test_nonfinite_evaluation_restores_modes_and_rejects_oversized_work():
    model = Probe().train()
    model.bias.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        evaluate_decoder_documents(model, (Document("a", b"test material"),))
    assert model.training
    with pytest.raises(ValueError, match="8192"):
        evaluate_decoder_documents(model, (Document("a", b"a" * 8194),))
