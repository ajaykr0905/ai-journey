"""Operate tagged dense GQA through the installed training and DDP workflows."""

from pathlib import Path

from ai_journey.ddp_acceptance import certify_crash_recovery, verify_crash_recovery
from ai_journey.gpt2_training_cli import replay_report, run_and_publish, verify_report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/day-19-demo-names.txt"
SPEC = ROOT / "config/gqa-cpu-lab.json"


def test_dense_gqa_cli_trains_verifies_and_replays_exact_evidence(tmp_path):
    report = tmp_path / "training.json"
    evidence = run_and_publish(
        source_path=SOURCE,
        spec_path=SPEC,
        checkpoint_path=tmp_path / "state.pt",
        report_path=report,
    )
    assert verify_report(report) == evidence
    assert replay_report(source_path=SOURCE, report_path=report) == evidence


def test_compact_gqa_checkpoint_recovers_after_actual_two_rank_crash(tmp_path):
    directory = tmp_path / "workers"
    report = certify_crash_recovery(source=SOURCE, spec=SPEC, directory=directory)
    assert report["world_size"] == 2
    assert report["metrics"]["lost_optimizer_updates"] == 1
    assert report["state_sha256"]["continuous"] == report["state_sha256"]["recovered"]
    assert verify_crash_recovery(report, directory=directory) == report
