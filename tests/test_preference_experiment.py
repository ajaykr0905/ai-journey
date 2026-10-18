import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from ai_journey.preference_data import PreferenceDataError
from ai_journey.preference_experiment import (
    PreferenceComparisonConfig,
    PREFERENCE_ARMS,
    PREFERENCE_LIMITATIONS,
    prepare_preference_context,
    preference_config_dict,
    preference_config_from_dict,
    run_preference_comparison,
    run_preference_fold,
    warm_preference_base,
)
from ai_journey.transformer_lab import model_fingerprint


def test_preference_config_round_trips_all_controls():
    config = PreferenceComparisonConfig()

    restored = preference_config_from_dict(preference_config_dict(config))

    assert restored == config
    assert isinstance(preference_config_dict(config)["seeds"], list)
    assert config.work_units <= 80_000_000


def test_preference_config_builds_a_bounded_layer_norm_decoder():
    config = PreferenceComparisonConfig()

    model = config.model_config(53)

    assert model.vocab_size == 53
    assert model.block_size == config.block_size
    assert model.normalization_mode == "layer_norm"
    assert model.dropout == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": (44,)},
        {"seeds": (44, 44)},
        {"fold_count": 1},
        {"block_size": 4},
        {"embedding_dim": 10, "head_count": 4},
        {"preference_steps": 0},
        {"beta": 0},
        {"preference_learning_rate": float("nan")},
    ],
)
def test_preference_config_rejects_invalid_controls(changes):
    values = asdict(PreferenceComparisonConfig())
    values.update(changes)

    with pytest.raises(PreferenceDataError):
        PreferenceComparisonConfig(**values)


def test_preference_config_rejects_unknown_fields():
    value = preference_config_dict(PreferenceComparisonConfig())
    value["ignored"] = True

    with pytest.raises(PreferenceDataError, match="exactly"):
        preference_config_from_dict(value)


def test_preference_context_binds_folds_sft_and_shared_vocabulary():
    context = prepare_preference_context(PreferenceComparisonConfig())
    evidence = context.evidence()

    assert len(context.folds) == 3
    assert evidence["preference_fixture"]["pairs"] == 12
    assert evidence["sft_fixture_id"] == "authored-weather-tools-v1"
    assert evidence["sft_fixture"]["train"]["conversation_ids"]
    assert evidence["folds"]["sha256"]
    assert evidence["vocabulary"] == list(context.codec.tokens)


def test_sft_warmup_is_exact_and_restores_caller_runtime_state():
    config = PreferenceComparisonConfig()
    context = prepare_preference_context(config)
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    first, first_evidence = warm_preference_base(config, context, seed=44)
    second, second_evidence = warm_preference_base(config, context, seed=44)

    assert model_fingerprint(first) == model_fingerprint(second)
    assert first_evidence == second_evidence
    assert first_evidence["initial_sha256"] != first_evidence["final_sha256"]
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])


def test_fold_run_compares_matched_full_and_low_rank_policies():
    config = PreferenceComparisonConfig(sft_steps=1, preference_steps=1)
    context = prepare_preference_context(config)
    base, _ = warm_preference_base(config, context, seed=44)
    before = model_fingerprint(base)

    result = run_preference_fold(base, context.folds[0], config, context, seed=44)

    assert model_fingerprint(base) == before
    assert result["fold"] == context.folds[0].evidence()
    assert result["schedule"]["steps"] == 1
    assert [trial["name"] for trial in result["trials"]] == list(PREFERENCE_ARMS)
    for trial in result["trials"]:
        assert trial["initial_function_max_abs_error"] == 0
        assert trial["frozen_before_sha256"] == trial["frozen_after_sha256"]
        assert trial["heldout"]["pairs"] == len(context.folds[0].heldout)
        assert (
            trial["training"]["reference_before_sha256"] == result["reference_sha256"]
        )
        assert trial["training"]["reference_after_sha256"] == result["reference_sha256"]
    assert result["trials"][0]["training"]["optimizer_steps"] == 0
    assert result["trials"][1]["training"]["optimizer_steps"] == 1
    for trial in result["trials"][2:]:
        assert trial["merge"]["probe_max_abs_error"] <= 1e-10
        assert (
            trial["parameters"]["trainable"]
            < result["trials"][1]["parameters"]["trainable"]
        )


def test_complete_comparison_runs_every_seed_fold_and_arm_exactly():
    config = PreferenceComparisonConfig(fold_count=2, sft_steps=1, preference_steps=1)

    first = run_preference_comparison(config)
    second = run_preference_comparison(config)

    assert first == second
    assert first["config"] == preference_config_dict(config)
    assert first["limitations"] == PREFERENCE_LIMITATIONS
    assert first["runtime"]["device"] == "cpu"
    assert first["runtime"]["dtype"] == "float64"
    assert first["runtime"]["threads"] == 1
    assert [row["seed"] for row in first["seeds"]] == list(config.seeds)
    assert sum(
        len(fold["trials"]) for seed in first["seeds"] for fold in seed["folds"]
    ) == len(config.seeds) * config.fold_count * len(PREFERENCE_ARMS)
    assert all(
        seed["warmup"]["final_sha256"] == seed["source_after_sha256"]
        for seed in first["seeds"]
    )
