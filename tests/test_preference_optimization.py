import math

import pytest
import torch

from ai_journey.preference_data import (
    PreferenceDataError,
    authored_preference_fixture,
    candidate_targets,
    preference_codec,
)
from ai_journey.preference_optimization import (
    candidate_mean_logprob,
    direct_preference_objective,
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


def test_preference_objective_is_log_two_at_reference_policy():
    chosen = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)

    objective = direct_preference_objective(
        chosen,
        rejected,
        chosen.detach(),
        rejected.detach(),
        beta=0.2,
    )

    assert float(objective.loss.detach()) == pytest.approx(math.log(2), abs=1e-12)
    assert float(objective.reward_margin.detach()) == 0


def test_preference_objective_pushes_chosen_up_and_rejected_down():
    chosen = torch.tensor(-1.5, dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)

    objective = direct_preference_objective(
        chosen,
        rejected,
        torch.tensor(-1.5, dtype=torch.float64),
        torch.tensor(-2.0, dtype=torch.float64),
        beta=1.0,
    )
    objective.loss.backward()

    assert float(chosen.grad) < 0
    assert float(rejected.grad) > 0


@pytest.mark.parametrize("beta", [0, -1, float("nan"), True, 11])
def test_preference_objective_rejects_invalid_beta(beta):
    value = torch.tensor(0.0)

    with pytest.raises(PreferenceDataError, match="beta"):
        direct_preference_objective(value, value, value, value, beta=beta)
