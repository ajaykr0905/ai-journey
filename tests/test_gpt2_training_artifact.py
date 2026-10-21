from hashlib import sha256
from pathlib import Path

from ai_journey.gpt2_training_evidence_io import load_training_evidence


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "day-19-demo-names.txt"
REPORT = ROOT / "artifacts" / "gpt2-training-2026-10-21" / "training-evidence.json"


def test_checked_in_training_evidence_passes_integrity_and_accounting():
    report = load_training_evidence(REPORT)

    assert report["parameter_count"] == 7_536
    assert report["progress"] == {"started_step": 0, "completed_step": 12}
    assert len(report["metrics"]) == 12
    assert sum(metric["target_tokens"] for metric in report["metrics"]) == 768
    assert all(metric["micro_batches"] == 4 for metric in report["metrics"])
    assert all(
        metric["gradient_norm_after_clip"] <= 1.0 for metric in report["metrics"]
    )
    assert report["metrics"][0]["mean_loss"] > report["metrics"][-1]["mean_loss"]


def test_checked_in_training_evidence_binds_the_public_source_and_runtime():
    report = load_training_evidence(REPORT)

    assert report["spec"]["source"]["sha256"] == sha256(SOURCE.read_bytes()).hexdigest()
    assert report["runtime"]["operating_system"]
    assert report["runtime"]["machine"]
    assert len(report["runtime"]["torch_config_sha256"]) == 64
