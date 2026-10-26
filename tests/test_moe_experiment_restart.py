import json

import pytest

from ai_journey.decoder_variant_comparison import decoder_comparison_digest
from ai_journey.moe_experiment_comparison import run_moe_document_comparison
from ai_journey.moe_experiment_restart import run_restarting_moe_comparison
import ai_journey.moe_experiment_restart as restart
from test_moe_experiment import TRAIN, HELDOUT
from test_moe_experiment_plan import plan


def test_interrupted_seed_restarts_while_completed_owned_artifact_stays_immutable(
    tmp_path, monkeypatch
):
    controls = plan(seeds=(53, 54))
    runner = restart.run_moe_document_trial

    def fail_second(*args, seed):
        if seed == 54:
            raise RuntimeError("injected interruption")
        return runner(*args, seed=seed)

    monkeypatch.setattr(restart, "run_moe_document_trial", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        run_restarting_moe_comparison(controls, TRAIN, HELDOUT, tmp_path)
    completed = tmp_path / "seed-0-53.json"
    original = completed.read_bytes()
    called = []

    def track(*args, seed):
        called.append(seed)
        return runner(*args, seed=seed)

    monkeypatch.setattr(restart, "run_moe_document_trial", track)
    actual = run_restarting_moe_comparison(controls, TRAIN, HELDOUT, tmp_path)
    assert called == [54]
    assert completed.read_bytes() == original
    assert actual == run_moe_document_comparison(controls, TRAIN, HELDOUT)
    monkeypatch.setattr(
        restart,
        "run_moe_document_trial",
        lambda *a, **k: pytest.fail("complete seeds must not rerun"),
    )
    assert run_restarting_moe_comparison(controls, TRAIN, HELDOUT, tmp_path) == actual


@pytest.mark.parametrize(
    "mutation", ["source", "re-sealed-step", "runtime", "hole", "symlink"]
)
def test_invalid_completed_evidence_rejects_before_new_training(
    tmp_path, monkeypatch, mutation
):
    controls = plan(seeds=(53, 54))
    run_restarting_moe_comparison(controls, TRAIN, HELDOUT, tmp_path)
    checkpoint = tmp_path / "seed-0-53.json"
    source = TRAIN
    progress = tmp_path
    if mutation == "source":
        source += b"changed source\n"
    elif mutation == "hole":
        checkpoint.unlink()
    elif mutation == "symlink":
        progress = tmp_path / "linked"
        progress.symlink_to(tmp_path, target_is_directory=True)
    else:
        payload = json.loads(checkpoint.read_text())
        if mutation == "runtime":
            payload["torch_version"] = "other-runtime"
        else:
            payload["trial"]["arms"][1]["trace"][0]["step"] = True
        payload["report_sha256"] = decoder_comparison_digest(payload)
        checkpoint.write_text(json.dumps(payload))
    monkeypatch.setattr(
        restart,
        "run_moe_document_trial",
        lambda *a, **k: pytest.fail("must reject before new training"),
    )
    with pytest.raises(ValueError):
        run_restarting_moe_comparison(controls, source, HELDOUT, progress)
