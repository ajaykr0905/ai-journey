import math
import copy

import pytest
import torch

from ai_journey.preference_data import (
    PreferenceDataError,
    authored_preference_fixture,
    candidate_targets,
    preference_codec,
    preference_training_batches,
)
from ai_journey.preference_optimization import (
    candidate_mean_logprob,
    direct_preference_objective,
    evaluate_candidate_logprob,
    evaluate_preferences,
    train_preference_batches,
)
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
)


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


def test_preference_training_updates_policy_and_preserves_reference():
    policy, _, codec = decoder_and_data()
    reference = copy.deepcopy(policy).requires_grad_(False)
    pairs = authored_preference_fixture().pairs
    batches = preference_training_batches(pairs, batch_size=2, steps=2, seed=44)
    policy_before = model_fingerprint(policy)
    reference_before = model_fingerprint(reference)

    trace = train_preference_batches(
        policy, reference, batches, codec, learning_rate=0.01, beta=0.2
    )

    assert model_fingerprint(policy) != policy_before
    assert model_fingerprint(reference) == reference_before
    assert trace["steps"] == 2
    assert trace["comparisons"] == 4
    assert trace["optimizer_steps"] == 2
    assert trace["reference_before_sha256"] == trace["reference_after_sha256"]
    assert all(math.isfinite(row["loss"]) for row in trace["trace"])


def test_frozen_preference_control_runs_without_updates():
    source, _, codec = decoder_and_data()
    policy = copy.deepcopy(source).requires_grad_(False)
    reference = copy.deepcopy(source).requires_grad_(False)
    batches = preference_training_batches(
        authored_preference_fixture().pairs, batch_size=2, steps=2, seed=44
    )
    before = model_fingerprint(policy)

    trace = train_preference_batches(
        policy, reference, batches, codec, learning_rate=0.01, beta=0.2
    )

    assert model_fingerprint(policy) == before
    assert trace["optimizer_steps"] == 0
    assert all(row["gradient_norm"] == 0 for row in trace["trace"])


def test_preference_training_restores_mixed_module_modes():
    policy, _, codec = decoder_and_data()
    reference = copy.deepcopy(policy).requires_grad_(False)
    policy.train()
    policy.blocks[0].eval()
    reference.eval()
    reference.blocks[0].train()
    policy_modes = [module.training for module in policy.modules()]
    reference_modes = [module.training for module in reference.modules()]
    batches = preference_training_batches(
        authored_preference_fixture().pairs, batch_size=1, steps=1, seed=44
    )

    train_preference_batches(
        policy, reference, batches, codec, learning_rate=0.01, beta=0.2
    )

    assert [module.training for module in policy.modules()] == policy_modes
    assert [module.training for module in reference.modules()] == reference_modes


def test_preference_evaluation_covers_each_pair_once():
    policy, _, codec = decoder_and_data()
    reference = copy.deepcopy(policy).requires_grad_(False)
    pairs = authored_preference_fixture().pairs[:4]

    result = evaluate_preferences(policy, reference, pairs, codec, beta=0.2)

    assert result["pairs"] == 4
    assert [row["pair_id"] for row in result["rows"]] == [
        pair.pair_id for pair in pairs
    ]
    assert 0 <= result["preference_accuracy"] <= 1
    assert result["mean_reward_margin"] == pytest.approx(0, abs=1e-12)
    assert result["mean_loss"] == pytest.approx(math.log(2), abs=1e-12)
    assert all(
        math.isfinite(result[name])
        for name in (
            "mean_policy_margin",
            "mean_reference_margin",
            "mean_reward_margin",
            "mean_loss",
        )
    )


def test_preference_evaluation_restores_modes():
    policy, _, codec = decoder_and_data()
    reference = copy.deepcopy(policy).requires_grad_(False)
    policy.train()
    policy.blocks[0].eval()
    reference.eval()
    reference.blocks[0].train()
    modules = [*policy.modules(), *reference.modules()]
    before = [module.training for module in modules]

    evaluate_preferences(
        policy,
        reference,
        authored_preference_fixture().pairs[:2],
        codec,
        beta=0.2,
    )

    assert [module.training for module in modules] == before
