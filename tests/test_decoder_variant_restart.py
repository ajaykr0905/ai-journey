import json

import pytest

from ai_journey.decoder_variant_comparison import run_decoder_document_comparison
from ai_journey.decoder_variant_restart import run_restarting_decoder_comparison
import ai_journey.decoder_variant_restart as restart
from test_decoder_variant_document_experiment import TRAIN, HELDOUT
from test_decoder_variant_plan import plan


def test_interrupted_run_reuses_completed_seeds_and_matches_fresh_evidence(
    tmp_path, monkeypatch
):
    controls = plan(seeds=(7, 13))
    actual_runner = restart.run_decoder_document_trial

    def fail_second(plan, train, heldout, *, seed):
        if seed == 13:
            raise RuntimeError("injected interruption")
        return actual_runner(plan, train, heldout, seed=seed)

    monkeypatch.setattr(restart, "run_decoder_document_trial", fail_second)
    with pytest.raises(RuntimeError, match="interruption"):
        run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, tmp_path)
    completed = tmp_path / "seed-0-7.json"
    original = completed.read_bytes()
    called = []

    def track(plan, train, heldout, *, seed):
        called.append(seed)
        return actual_runner(plan, train, heldout, seed=seed)

    monkeypatch.setattr(restart, "run_decoder_document_trial", track)
    resumed = run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, tmp_path)
    assert called == [13]
    assert completed.read_bytes() == original
    assert resumed == run_decoder_document_comparison(controls, TRAIN, HELDOUT)
    called.clear()
    assert (
        run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, tmp_path) == resumed
    )
    assert not called


def test_restart_rejects_changed_inputs_corruption_and_symlinked_progress(tmp_path):
    controls = plan()
    run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, tmp_path)
    checkpoint = tmp_path / "seed-0-52.json"
    with pytest.raises(ValueError, match="identity"):
        run_restarting_decoder_comparison(
            controls, TRAIN + b"changed", HELDOUT, tmp_path
        )
    payload = json.loads(checkpoint.read_text())
    payload["trial"]["arms"][0]["trace"][0]["loss"] = 0
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_restarting_decoder_comparison(controls, TRAIN, HELDOUT, linked)
