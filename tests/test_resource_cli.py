from dataclasses import asdict
import json
from pathlib import Path

from ai_journey.resource_cli import main, verify_main
from ai_journey.resource_report_io import load_resource_report
from test_resource_report import make_plan


def test_resource_cli_builds_verified_report(tmp_path: Path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(asdict(make_plan())), encoding="utf-8")

    assert main(["--config", str(config), "--output", str(output)]) == 0

    report = load_resource_report(output)
    assert report["parameters"]["count"] == 7_888
    stdout = capsys.readouterr().out
    assert "fits=true" in stdout
    assert "measured=false" in stdout

    assert verify_main(["--report", str(output)]) == 0
    verified = capsys.readouterr().out
    assert report["integrity"]["digest"] in verified
    assert "measured=false" in verified
