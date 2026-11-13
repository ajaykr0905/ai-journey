import json

from test_recovery_config import valid_config

from ai_journey.recovery_experiment import run_recovery_experiment
from ai_journey.recovery_report import build_recovery_report


def test_report_seals_measured_recovery_and_explicit_claim_boundary(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"public recovery fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(valid_config().to_mapping()))
    result = run_recovery_experiment(
        source_path=source, config_path=config, run_dir=tmp_path / "run"
    )

    report = build_recovery_report(result)

    assert report == build_recovery_report(result)
    assert report["experiment"]["work_items"] == 20
    assert report["attempts"]["failed_exit_code"] != 0
    assert report["attempts"]["resumed_exit_code"] == 0
    assert report["fault"]["mechanism"] == "process_kill"
    assert report["loss_assessment"]["lost_uncheckpointed_items"] == 3
    assert report["loss_assessment"]["omitted_items"] == 0
    assert report["timing"]["recovery_time_ms"] > 0
    assert "local CPU" in report["evidence_boundary"]
    assert "does not prove NVIDIA GPU" in report["evidence_boundary"]
    assert len(report["report_sha256"]) == 64
