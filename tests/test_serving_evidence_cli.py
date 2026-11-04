import json
from pathlib import Path

from test_serving_report import _report

from ai_journey.serving_evidence_cli import main
from ai_journey.serving_report import seal_serving_report
from ai_journey.serving_report_io import write_sealed_report


def test_serving_evidence_cli_emits_verified_receipt(tmp_path: Path, capsys):
    path = tmp_path / "report.json"
    sealed = seal_serving_report(_report())
    write_sealed_report(path, sealed)

    assert main([str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "engine_label": "fixture-engine",
        "execution_kind": "cpu_fixture",
        "report_sha256": sealed.report_sha256,
        "verified": True,
    }


def test_serving_evidence_cli_fails_closed_for_tampered_input(tmp_path: Path, capsys):
    path = tmp_path / "report.json"
    sealed = seal_serving_report(_report())
    write_sealed_report(path, sealed)
    document = json.loads(path.read_text())
    document["report"]["snapshot"]["peak_memory_bytes"] += 1
    path.write_text(json.dumps(document))

    assert main([str(path)]) == 1
    assert "digest mismatch" in capsys.readouterr().err
