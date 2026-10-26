import copy
import json

import pytest

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_experiment_comparison import run_moe_document_comparison
from ai_journey.moe_report import verify_moe_report
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


@pytest.fixture(scope="module")
def measured():
    return run_moe_document_comparison(plan(seeds=(53, 54)), TRAIN, HELDOUT)


def test_offline_verification_recomputes_actual_evidence_without_model_execution(
    measured, monkeypatch
):
    import ai_journey.training_models as models

    monkeypatch.setattr(
        models,
        "build_training_model",
        lambda *a, **k: pytest.fail("offline verification must not allocate models"),
    )
    actual = json.loads(json.dumps(measured))
    assert verify_moe_report(actual) == plan(seeds=(53, 54))
    assert "cannot authenticate" in actual["verification_scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime-alias",
        "step-alias",
        "parameter-alias",
        "objective",
        "dense-aux",
        "flag",
        "count",
        "loss",
        "mask",
        "summary",
        "shared",
        "clip-alias",
        "source",
        "unknown",
        "probability-alias",
        "route-shape",
    ],
)
def test_resealed_impossible_or_aliased_evidence_is_rejected(measured, mutation):
    report = copy.deepcopy(measured)
    trial = report["trials"][0]
    sparse = trial["arms"][1]
    row = sparse["trace"][0]
    if mutation == "runtime-alias":
        report["runtime"]["threads"] = True
    elif mutation == "step-alias":
        row["step"] = True
    elif mutation == "parameter-alias":
        sparse["parameter_count"] = float(sparse["parameter_count"])
    elif mutation == "objective":
        row["objective"] += 0.1
    elif mutation == "dense-aux":
        trial["arms"][0]["trace"][0]["balance_loss"] = 1.0
    elif mutation == "flag":
        row["routing"][0]["accepted_routes"][0][0] = False
    elif mutation == "count":
        row["routing"][0]["accepted_counts"][0] += 1
    elif mutation == "loss":
        sparse["heldout"]["batches"][0]["token_nll"][0] += 0.1
    elif mutation == "mask":
        sparse["heldout"]["batches"][0]["token_mask"][0][0] = False
    elif mutation == "summary":
        report["summary"]["paired_deltas"][0]["per_seed"][0]["delta_nll"] += 0.01
    elif mutation == "shared":
        sparse["initial_shared_sha256"] = "0" * 64
    elif mutation == "clip-alias":
        row["clipped"] = int(row["clipped"])
    elif mutation == "source":
        report["trials"][1]["train_sha256"] = "0" * 64
    elif mutation == "unknown":
        sparse["heldout"]["extra_note"] = "not an executable field"
    elif mutation == "probability-alias":
        row["routing"][0]["probability_sums"][0] = 0
    else:
        row["routing"] = ["invalid"]
    report["report_sha256"] = decoder_comparison_digest(report)
    with pytest.raises(ValueError):
        verify_moe_report(report)


def test_stale_seal_is_rejected_before_accepting_any_analytical_equality(measured):
    report = copy.deepcopy(measured)
    report["runtime"]["threads"] = 1.0
    with pytest.raises(ValueError, match="digest"):
        verify_moe_report(report)


def test_offline_contract_retains_recorded_runtime_while_replay_is_separate(measured):
    report = copy.deepcopy(measured)
    report["runtime"]["torch_version"] = "recorded-other-version"
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_moe_report(report)
