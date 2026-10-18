import copy

import pytest

from ai_journey.preference_data import PreferenceDataError
from ai_journey.preference_experiment import (
    PREFERENCE_ARMS,
    PreferenceComparisonConfig,
    run_preference_comparison,
)
from ai_journey.preference_report import summarize_preference_comparison
from ai_journey.preference_report import (
    PREFERENCE_REPORT_SCHEMA,
    build_preference_report,
    preference_report_digest,
    validate_preference_report,
)


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


def test_report_binds_protocol_sources_summary_and_digest(comparison):
    report = build_preference_report(comparison)

    assert report["schema"] == PREFERENCE_REPORT_SCHEMA
    assert report["sha256"] == preference_report_digest(report)
    assert validate_preference_report(report) is report
    assert report["summary"] == summarize_preference_comparison(report["result"])


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema",), "wrong", "schema"),
        (("sha256",), "0" * 64, "digest"),
        (("result", "context", "preference_fixture", "pairs"), 11, "context"),
        (("result", "runtime", "threads"), 2, "runtime controls"),
        (("summary", "observations"), 99, "summary"),
    ],
)
def test_report_rejects_tampered_owned_evidence(comparison, path, value, match):
    report = build_preference_report(comparison)
    current = report
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    if path != ("sha256",):
        report["sha256"] = preference_report_digest(report)

    with pytest.raises(PreferenceDataError, match=match):
        validate_preference_report(report)
