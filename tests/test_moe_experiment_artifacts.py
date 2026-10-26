from dataclasses import replace
import json
import os

import pytest
import torch

import ai_journey.moe_experiment_artifacts as artifacts
from ai_journey.moe_cli import main
from ai_journey.moe_evaluation import evaluate_moe_documents
from ai_journey.moe_experiment_artifacts import (
    load_moe_experiment_artifact,
    publish_moe_experiment,
)
from ai_journey.moe_experiment_comparison import run_moe_document_comparison
from ai_journey.decoder_variant_data import prepare_decoder_documents
from ai_journey.training_models import training_model_fingerprint
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


def test_published_state_matches_unchanged_report_and_actual_heldout_and_cache(
    tmp_path,
):
    controls = plan(seeds=(53, 54))
    controls = replace(
        controls,
        arms=tuple(
            replace(arm, config=replace(arm.config, position_kind="rope"))
            for arm in controls.arms
        ),
    )
    rng = torch.get_rng_state().clone()
    destination = tmp_path / "trained"
    index = publish_moe_experiment(
        controls, TRAIN, HELDOUT, destination, arm="top2", seed=54
    )
    model, report = load_moe_experiment_artifact(destination, TRAIN)
    assert torch.equal(torch.get_rng_state(), rng)
    assert report == run_moe_document_comparison(controls, TRAIN, HELDOUT)
    measured = report["trials"][1]["arms"][1]
    assert training_model_fingerprint(model) == measured["final_model_sha256"]
    assert index["report_sha256"] == report["report_sha256"]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    documents = prepare_decoder_documents(TRAIN, HELDOUT, context_length=4)
    assert evaluate_moe_documents(model, documents.heldout) == measured["heldout"]
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        first, cache = model.forward_with_cache(ids[:, :2])
        second, _ = model.forward_with_cache(ids[:, 2:], past_key_values=cache)
        torch.testing.assert_close(torch.cat((first, second), 1), model(ids))


@pytest.fixture
def published(tmp_path):
    destination = tmp_path / "trained"
    publish_moe_experiment(plan(), TRAIN, HELDOUT, destination, arm="top2", seed=53)
    return destination


@pytest.mark.parametrize(
    "mutation", ["seed_alias", "arm", "contract", "report", "unknown"]
)
def test_resealed_index_cannot_rebind_measured_state(published, mutation):
    path = published / "index.json"
    index = json.loads(path.read_text())
    if mutation == "seed_alias":
        index["seed"] = True
    elif mutation == "arm":
        index["arm"] = "baseline"
    elif mutation == "contract":
        index["training_contract"]["seed"] = 54
    elif mutation == "report":
        index["report_sha256"] = "0" * 64
    else:
        index["extra"] = "unverified"
    index["index_sha256"] = artifacts._index_digest(index)
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        load_moe_experiment_artifact(published, TRAIN)


def test_changed_source_or_checkpoint_bytes_are_rejected(published):
    with pytest.raises(ValueError, match="source binding"):
        load_moe_experiment_artifact(published, TRAIN + b"changed")
    with (published / "model.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file digest"):
        load_moe_experiment_artifact(published, TRAIN)


def test_resealed_report_final_fingerprint_must_match_actual_checkpoint(published):
    from ai_journey.decoder_variant_comparison import decoder_comparison_digest

    report_path = published / "report.json"
    report = json.loads(report_path.read_text())
    report["trials"][0]["arms"][1]["final_model_sha256"] = "0" * 64
    report["report_sha256"] = decoder_comparison_digest(report)
    report_path.write_text(json.dumps(report))
    path = published / "index.json"
    index = json.loads(path.read_text())
    index["report_sha256"] = report["report_sha256"]
    index["index_sha256"] = artifacts._index_digest(index)
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="measured final arm"):
        load_moe_experiment_artifact(published, TRAIN)


def test_existing_or_symlink_outputs_are_never_overwritten(published, monkeypatch):
    original = {p.name: p.read_bytes() for p in published.iterdir()}
    monkeypatch.setattr(
        artifacts, "run_moe_document_trial", lambda *a, **kw: pytest.fail("no training")
    )
    with pytest.raises(ValueError, match="already exists"):
        publish_moe_experiment(plan(), TRAIN, HELDOUT, published, arm="top2", seed=53)
    assert original == {p.name: p.read_bytes() for p in published.iterdir()}
    link = published.parent / "link"
    link.symlink_to(published, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        load_moe_experiment_artifact(link, TRAIN)


def test_failed_training_does_not_claim_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(
        artifacts,
        "run_moe_document_trial",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        publish_moe_experiment(
            plan(), TRAIN, HELDOUT, tmp_path / "trained", arm="top2", seed=53
        )
    assert list(tmp_path.iterdir()) == []


def test_interrupted_publication_has_no_commit_marker_and_is_not_reused(
    tmp_path, monkeypatch
):
    real_link = os.link

    def fail_index(source, target):
        if target.parent.name == "trained" and target.name == "index.json":
            raise OSError("injected publication interruption")
        return real_link(source, target)

    monkeypatch.setattr(artifacts.os, "link", fail_index)
    destination = tmp_path / "trained"
    with pytest.raises(OSError, match="interruption"):
        publish_moe_experiment(plan(), TRAIN, HELDOUT, destination, arm="top2", seed=53)
    assert destination.exists() and not (destination / "index.json").exists()
    with pytest.raises(ValueError, match="complete"):
        load_moe_experiment_artifact(destination, TRAIN)
    with pytest.raises(ValueError, match="already exists"):
        publish_moe_experiment(plan(), TRAIN, HELDOUT, destination, arm="top2", seed=53)


@pytest.mark.parametrize(
    "arm,seed", [("baseline", 53), ("missing", 53), ("top2", True), ("top2", 54)]
)
def test_invalid_selection_rejects_before_execution(tmp_path, monkeypatch, arm, seed):
    monkeypatch.setattr(
        artifacts,
        "run_moe_document_trial",
        lambda *a, **kw: pytest.fail("must not run"),
    )
    with pytest.raises(ValueError):
        publish_moe_experiment(
            plan(), TRAIN, HELDOUT, tmp_path / "trained", arm=arm, seed=seed
        )


def test_cli_publishes_and_loads_actual_trained_artifact(tmp_path, capsys):
    train, heldout, config = (tmp_path / name for name in ("train", "heldout", "plan"))
    train.write_bytes(TRAIN)
    heldout.write_bytes(HELDOUT)
    config.write_text(json.dumps(plan().to_mapping()))
    destination = tmp_path / "trained"
    assert (
        main(
            [
                "publish",
                "--plan",
                str(config),
                "--train",
                str(train),
                "--heldout",
                str(heldout),
                "--directory",
                str(destination),
                "--arm",
                "top2",
                "--seed",
                "53",
            ]
        )
        == 0
    )
    assert (
        main(
            ["artifact-verify", "--directory", str(destination), "--train", str(train)]
        )
        == 0
    )
    assert "frozen=True" in capsys.readouterr().out
