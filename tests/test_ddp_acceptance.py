import json
from pathlib import Path
from ai_journey.ddp_acceptance import certify_recovery
from ai_journey.ddp_acceptance import certify_batch_equivalence


def test_two_rank_restart_reproduces_model_optimizer_rng_and_progress(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=4, sequence_length=4, micro_batch_size=1, accumulation_steps=2
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    result = certify_recovery(
        source=root / "data/day-19-demo-names.txt",
        spec=spec,
        directory=tmp_path / "recovery",
    )
    assert result["exact_state"] == {
        "model": True,
        "optimizer": True,
        "rank_rngs": True,
        "step": True,
    }
    assert result["completed_step"] == 4


def test_ddp_matches_one_process_with_the_same_global_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config/gpt2-training.json").read_text())
    payload["training"].update(
        optimizer_steps=2, sequence_length=4, micro_batch_size=1, accumulation_steps=2
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(payload))
    result = certify_batch_equivalence(
        source=root / "data/day-19-demo-names.txt",
        spec=spec,
        directory=tmp_path / "equivalence",
    )
    assert result["equivalent"]
    assert result["global_batch_size"] == 4
    assert result["max_parameter_error"] < 1e-10
