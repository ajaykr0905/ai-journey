from hashlib import sha256
import json
from pathlib import Path
from ai_journey.ddp_comparison import verify_comparison
from ai_journey.gpt2_training import load_training_spec


ROOT = Path(__file__).resolve().parents[1]


def test_published_protocol_and_comparison_bind_the_public_source():
    spec = load_training_spec(ROOT / "config/ddp-lab.json")
    assert spec.training.seed == 48
    assert (
        spec.source.sha256
        == sha256((ROOT / "data/day-19-demo-names.txt").read_bytes()).hexdigest()
    )
    report = verify_comparison(
        json.loads((ROOT / "artifacts/ddp-lab-2026-10-22/comparison.json").read_text())
    )
    assert all(
        arm["evidence"]["source_sha256"] == spec.source.sha256 for arm in report["arms"]
    )
    assert all(arm["evidence"]["completed_step"] == 4 for arm in report["arms"])
    assert max(report["max_parameter_errors"]) < 1e-10
