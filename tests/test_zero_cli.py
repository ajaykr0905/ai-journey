from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ai_journey.zero_cli import main
from ai_journey.zero_io import load_zero_comparison_report
from ai_journey.zero_report import ZeroReportError
from test_zero_report import _plan


def test_zero_cli_builds_and_verifies_recomputable_evidence(tmp_path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(asdict(_plan())), encoding="utf-8")

    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    build_output = capsys.readouterr().out
    assert "measured=false" in build_output
    assert "stages=4" in build_output
    report = load_zero_comparison_report(output)
    assert [stage["stage"] for stage in report["stages"]] == [0, 1, 2, 3]

    assert main(["verify", "--report", str(output)]) == 0
    assert "verified_zero=" in capsys.readouterr().out


def test_zero_cli_rejects_config_output_alias(tmp_path):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(asdict(_plan())), encoding="utf-8")

    with pytest.raises(ZeroReportError, match="must not alias"):
        main(["build", "--config", str(config), "--output", str(config)])


def test_zero_cli_rejects_duplicate_plan_fields(tmp_path):
    config = tmp_path / "plan.json"
    config.write_text('{"model":{},"model":{}}', encoding="utf-8")

    with pytest.raises(ZeroReportError, match="duplicate"):
        main(
            [
                "build",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_pinned_day57_plan_builds_public_safe_stage_comparison(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    config = repository / "config" / "day-57-zero-comparison.json"
    output = tmp_path / "zero-comparison.json"

    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    report = load_zero_comparison_report(output)

    assert report["topology"]["world_size"] == 8
    assert report["topology"]["workers_per_node"] == 8
    assert report["measured"] is False
    assert report["stages"][3]["max_peak_bytes"] < report["stages"][0]["max_peak_bytes"]
    assert [item["operation"] for item in report["stages"][3]["collectives"]] == [
        "all_gather",
        "reduce_scatter",
    ]
    assert [stage["admission"]["fits"] for stage in report["stages"]] == [
        False,
        True,
        True,
        True,
    ]
