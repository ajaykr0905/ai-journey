from dataclasses import asdict
import json
from pathlib import Path

from ai_journey.resource_accounting import SEVEN_B_DECODER
from ai_journey.resource_cli import load_resource_plan, main, verify_main
from ai_journey.resource_report import build_resource_report
from ai_journey.resource_report_io import load_resource_report
from test_resource_report import make_plan

ROOT = Path(__file__).resolve().parents[1]


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


def test_pinned_day_51_plan_matches_the_manual_exercise():
    plan = load_resource_plan(ROOT / "config" / "day-51-resource-plan.json")
    assert plan.model == SEVEN_B_DECODER
    assert plan.workload.tokens_per_optimizer_step == 8 * 2_048
    assert plan.workload.total_training_tokens == 16_384_000
    report = build_resource_report(plan)
    assert report["measured"] is False
    assert report["memory"]["capacity"]["fits"] is True
    assert report["cost"]["estimated_cost"] > 0
