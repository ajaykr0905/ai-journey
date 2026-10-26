import copy

import pytest

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.gqa_cli import default_gqa_plan
from ai_journey.gqa_experiment import run_gqa_comparison
from ai_journey.gqa_experiment_report import (
    verify_gqa_experiment_report,
    replay_gqa_experiment_report,
)
from test_moe_experiment import TRAIN, HELDOUT


@pytest.fixture(scope="module")
def measured():
    return run_gqa_comparison(default_gqa_plan(), TRAIN, HELDOUT)


@pytest.mark.parametrize(
    "mutation",
    [
        "alias",
        "parameter",
        "inventory",
        "loss",
        "window",
        "optimizer",
        "aux",
        "clip",
        "initialization",
        "summary",
        "budget",
        "seed",
    ],
)
def test_resealed_impossible_gqa_records_reject(measured, monkeypatch, mutation):
    import ai_journey.training_models as models

    report = copy.deepcopy(measured)
    arm = report["trials"][0]["arms"][1]
    if mutation == "alias":
        report["runtime"]["threads"] = True
    elif mutation == "parameter":
        arm["parameter_count"] += 1
    elif mutation == "inventory":
        arm["parameter_inventory"]["attention"] += 1
    elif mutation == "loss":
        arm["heldout"]["batches"][0]["token_nll"][0] += 0.1
    elif mutation == "window":
        arm["heldout"]["batches"][0]["token_mask"][0][0] = False
    elif mutation == "optimizer":
        arm["optimizer_groups"]["decay_names"][0] = "nonexistent.parameter"
    elif mutation == "aux":
        arm["trace"][0]["balance_loss"] = 1.0
    elif mutation == "clip":
        arm["trace"][0]["clipped"] = 1
    elif mutation == "initialization":
        arm["initialization"]["exact_shared_sha256"] = "0" * 64
    elif mutation == "summary":
        report["summary"]["paired_deltas"][0]["per_seed"][0]["delta_nll"] += 0.1
    elif mutation == "budget":
        report["resource_admission"]["work_units"] -= 1
    else:
        report["trials"][0]["seed"] = True
    report["report_sha256"] = decoder_comparison_digest(report)
    monkeypatch.setattr(
        models,
        "build_training_model",
        lambda *a, **kw: pytest.fail("offline must not allocate"),
    )
    with pytest.raises(ValueError):
        verify_gqa_experiment_report(report)


def test_internal_identity_is_not_authentication_and_requires_actual_replay(measured):
    report = copy.deepcopy(measured)
    report["trials"][0]["arms"][1]["final_model_sha256"] = "0" * 64
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_gqa_experiment_report(report)
    with pytest.raises(ValueError, match="actual source replay"):
        replay_gqa_experiment_report(report, TRAIN, HELDOUT)


def test_wrong_actual_source_rejects_before_execution(measured, monkeypatch):
    import ai_journey.gqa_experiment_report as verifier

    monkeypatch.setattr(
        verifier, "run_gqa_comparison", lambda *a: pytest.fail("no execution")
    )
    with pytest.raises(ValueError, match="source bytes"):
        replay_gqa_experiment_report(measured, TRAIN + b"changed", HELDOUT)


def test_resealed_gqa_gradient_disappearance_is_not_a_valid_clip_transition(measured):
    report = copy.deepcopy(measured)
    row = report["trials"][0]["arms"][1]["trace"][0]
    assert row["gradient_norm_before_clip"] > 0.1
    row["gradient_norm_after_clip"] = 0.0
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError, match="clipping"):
        verify_gqa_experiment_report(report)
