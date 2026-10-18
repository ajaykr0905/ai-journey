import math

import torch

from ai_journey.preference_data import (
    authored_preference_fixture,
    candidate_targets,
    preference_codec,
)
from ai_journey.preference_optimization import (
    candidate_mean_logprob,
    evaluate_candidate_logprob,
)
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


def decoder_and_data():
    dataset = authored_preference_fixture()
    codec = preference_codec(dataset)
    config = TransformerConfig(
        vocab_size=len(codec.tokens),
        block_size=16,
        embedding_dim=8,
        head_count=2,
        layer_count=1,
        dropout=0.0,
    )
    return DecoderLanguageModel(config).double(), dataset.pairs[0], codec


def test_candidate_mean_logprob_matches_one_target_at_a_time():
    model, pair, codec = decoder_and_data()
    targets = candidate_targets(pair, "chosen", codec, block_size=16)

    actual = candidate_mean_logprob(model, pair, "chosen", codec)
    expected = []
    for context, target in zip(targets.contexts, targets.targets, strict=True):
        logits, _ = model(torch.tensor([context], dtype=torch.long))
        expected.append(torch.log_softmax(logits[0, -1], dim=-1)[target])

    assert torch.allclose(actual, torch.stack(expected).mean(), atol=1e-12, rtol=0)
    assert actual.requires_grad


def test_candidate_evaluation_is_finite_and_restores_modes():
    model, pair, codec = decoder_and_data()
    model.train()
    model.blocks[0].eval()
    before = [module.training for module in model.modules()]

    value = evaluate_candidate_logprob(model, pair, "chosen", codec)

    assert math.isfinite(value)
    assert [module.training for module in model.modules()] == before


def test_chosen_and_rejected_candidates_have_distinct_scores():
    model, pair, codec = decoder_and_data()

    chosen = evaluate_candidate_logprob(model, pair, "chosen", codec)
    rejected = evaluate_candidate_logprob(model, pair, "rejected", codec)

    assert chosen != rejected
