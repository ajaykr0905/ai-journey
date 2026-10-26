import copy
import json

import pytest
import torch

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.gqa_cli import default_gqa_plan, main
from ai_journey.gqa_cache_evidence import (
    run_gqa_cache_evidence,
    verify_gqa_cache_evidence,
    replay_gqa_cache_evidence,
)
from ai_journey.gqa_experiment import run_gqa_comparison
from ai_journey.gqa_experiment_public import run_public_gqa_comparison
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_cli import snapshot


@pytest.fixture(scope="module")
def measured():
    source = run_gqa_comparison(default_gqa_plan(), TRAIN, HELDOUT)
    return run_gqa_cache_evidence(source, TRAIN, HELDOUT, prefix_count=2)


def test_actual_growth_retains_native_backing_stores_and_recomputable_logits(measured):
    verify_gqa_cache_evidence(measured)
    rng = torch.get_rng_state().clone()
    replay_gqa_cache_evidence(measured, TRAIN, HELDOUT)
    assert torch.equal(rng, torch.get_rng_state())
    summary = measured["summary"]
    assert [row["max_retained_cache_bytes"] for row in summary] == [512, 256, 128]
    assert len({row["compared_logits"] for row in summary}) == 1
    assert all(row["max_tolerance_ratio"] <= 1 for row in summary)
    for observation in measured["observations"]:
        for length, step in enumerate(observation["steps"], 1):
            assert step["shape"] == [1, observation["kv_heads"], length, 2]
            assert (
                step["storage"]["logical_bytes"]
                == step["storage"]["retained_storage_bytes"]
            )
            assert step["execution"]["score_elements"] == 4 * length
            assert step["parity"]["compared_logits"] == 256
    assert (
        measured["resource_admission"]["work_units"]
        > measured["source_report"]["resource_admission"]["work_units"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "storage",
        "expanded",
        "score",
        "logit",
        "residual",
        "alias",
        "growth",
        "prefix",
        "model",
        "summary",
        "budget",
        "precision",
    ],
)
def test_resealed_impossible_cache_evidence_rejects(measured, mutation, monkeypatch):
    import ai_journey.training_models as models

    report = copy.deepcopy(measured)
    row = report["observations"][1]
    step = row["steps"][0]
    if mutation == "storage":
        step["storage"]["retained_storage_bytes"] += 4
    elif mutation == "expanded":
        step["shape"][1] = 8
    elif mutation == "score":
        step["execution"]["score_elements"] //= 2
    elif mutation == "logit":
        step["cached_logits"][0] += 1
    elif mutation == "residual":
        step["parity"]["max_absolute_error"] += 0.1
    elif mutation == "alias":
        step["length"] = True
    elif mutation == "growth":
        row["steps"].pop()
    elif mutation == "prefix":
        row["prefix"]["sha256"] = "0" * 64
    elif mutation == "model":
        row["model_sha256"] = "0" * 64
    elif mutation == "summary":
        report["summary"][0]["compared_logits"] -= 1
    elif mutation == "budget":
        report["resource_admission"]["cache_audit_work_units"] -= 1
    else:
        step["full_logits"][0] += 1e-12
    report["report_sha256"] = decoder_comparison_digest(report)
    monkeypatch.setattr(
        models,
        "build_training_model",
        lambda *a, **kw: pytest.fail("offline must not allocate"),
    )
    with pytest.raises(ValueError):
        verify_gqa_cache_evidence(report)


def test_consistently_fabricated_logits_need_actual_model_replay(measured):
    import ai_journey.gqa_cache_evidence as evidence

    report = copy.deepcopy(measured)
    step = report["observations"][0]["steps"][0]
    step["full_logits"][0] = step["cached_logits"][0] = 0.0
    step["parity"] = evidence._parity(step["full_logits"], step["cached_logits"])
    report["summary"] = evidence._summarize(default_gqa_plan(), report["observations"])
    report["report_sha256"] = decoder_comparison_digest(report)
    verify_gqa_cache_evidence(report)
    with pytest.raises(ValueError, match="actual source/model replay"):
        replay_gqa_cache_evidence(report, TRAIN, HELDOUT)


def test_wrong_source_and_unsafe_complete_audit_budget_reject_before_allocation(
    measured, monkeypatch
):
    import ai_journey.gqa_cache_evidence as evidence

    monkeypatch.setattr(
        evidence, "run_gqa_comparison", lambda *a, **kw: pytest.fail("no allocation")
    )
    with pytest.raises(ValueError, match="original source"):
        run_gqa_cache_evidence(measured["source_report"], TRAIN + b"changed", HELDOUT)
    import ai_journey.gqa_experiment_plan as admission

    monkeypatch.setattr(
        admission,
        "MAX_EXPERIMENT_WORK_UNITS",
        measured["source_report"]["resource_admission"]["work_units"] + 1,
    )
    with pytest.raises(ValueError, match="work_units"):
        run_gqa_cache_evidence(
            measured["source_report"], TRAIN, HELDOUT, prefix_count=4
        )


def test_public_cli_cache_generation_verification_replay_and_no_overwrite(
    tmp_path, monkeypatch
):
    path = snapshot(tmp_path, monkeypatch)
    source = run_public_gqa_comparison(
        default_gqa_plan(), path, "en-pud", train_documents=4, heldout_documents=2
    )
    original, output = tmp_path / "source.json", tmp_path / "cache.json"
    original.write_text(json.dumps(source))
    assert (
        main(
            [
                "cache",
                "--report",
                str(original),
                "--snapshot",
                str(path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert main(["cache-verify", "--report", str(output)]) == 0
    assert main(["cache-replay", "--report", str(output), "--snapshot", str(path)]) == 0
    unchanged = output.read_bytes()
    with pytest.raises(SystemExit):
        main(
            [
                "cache",
                "--report",
                str(original),
                "--snapshot",
                str(path),
                "--output",
                str(output),
            ]
        )
    assert output.read_bytes() == unchanged
