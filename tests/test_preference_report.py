import copy

import pytest

from ai_journey.preference_data import PreferenceDataError
from ai_journey.preference_experiment import (
    PREFERENCE_ARMS,
    PreferenceComparisonConfig,
    run_preference_comparison,
)
from ai_journey.preference_report import summarize_preference_comparison


@pytest.fixture(scope="module")
def comparison():
    return run_preference_comparison(
        PreferenceComparisonConfig(fold_count=2, sft_steps=1, preference_steps=1)
    )


def test_summary_preserves_seed_fold_pairing_and_uncertainty(comparison):
    summary = summarize_preference_comparison(comparison)

    assert summary["observations"] == 4
    assert [arm["name"] for arm in summary["arms"]] == list(PREFERENCE_ARMS)
    assert len(summary["paired"]) == 6
    assert all(arm["preference_accuracy"]["count"] == 4 for arm in summary["arms"])
    assert all(len(row["observations"]) == 4 for row in summary["paired"])
    assert all(0 <= row["candidate_accuracy_wins"] <= 4 for row in summary["paired"])
    assert isinstance(summary["negative_findings"], list)


def test_summary_records_low_rank_trainable_fractions(comparison):
    summary = summarize_preference_comparison(comparison)
    arms = {arm["name"]: arm for arm in summary["arms"]}

    assert arms["full"]["trainable_fraction_of_full"] == 1
    assert arms["frozen"]["trainable_fraction_of_full"] == 0
    assert 0 < arms["lora-rank-1"]["trainable_fraction_of_full"] < 1
    assert (
        arms["lora-rank-1"]["trainable_fraction_of_full"]
        < arms["lora-rank-2"]["trainable_fraction_of_full"]
        < 1
    )


def test_summary_rejects_missing_fold_coverage(comparison):
    damaged = copy.deepcopy(comparison)
    damaged["seeds"][0]["folds"].pop()

    with pytest.raises(PreferenceDataError, match="every cross-validation fold"):
        summarize_preference_comparison(damaged)


def test_summary_rejects_inconsistent_parameter_counts(comparison):
    damaged = copy.deepcopy(comparison)
    damaged["seeds"][1]["folds"][0]["trials"][1]["parameters"]["trainable"] += 1

    with pytest.raises(PreferenceDataError, match="parameter counts"):
        summarize_preference_comparison(damaged)
