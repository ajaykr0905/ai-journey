from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ai_journey.distributed_capacity_cli import main
from ai_journey.distributed_capacity_io import load_distributed_capacity_report
from ai_journey.distributed_capacity_report import DistributedCapacityReportError
from test_distributed_capacity_report import _plan


def test_capacity_cli_builds_and_verifies_recomputable_report(tmp_path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(asdict(_plan())), encoding="utf-8")

    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    assert "measured=false" in capsys.readouterr().out
    assert main(["verify", "--report", str(output)]) == 0
    assert "verified_capacity=" in capsys.readouterr().out


def test_capacity_cli_rejects_config_output_alias(tmp_path):
    config = tmp_path / "plan.json"
    config.write_text(json.dumps(asdict(_plan())), encoding="utf-8")
    with pytest.raises(DistributedCapacityReportError, match="must not alias"):
        main(["build", "--config", str(config), "--output", str(config)])


def test_capacity_cli_rejects_duplicate_plan_fields(tmp_path):
    config = tmp_path / "plan.json"
    config.write_text('{"model":{},"model":{}}', encoding="utf-8")
    with pytest.raises(DistributedCapacityReportError, match="duplicate"):
        main(
            [
                "build",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_pinned_day55_capacity_plan_builds_public_safe_analytical_evidence(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    config = repository / "config" / "day-55-distributed-capacity.json"
    output = tmp_path / "capacity.json"
    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    report = load_distributed_capacity_report(output)
    assert report["topology"]["world_size"] == 16
    assert report["topology"]["workers_per_node"] == 8
    assert report["gradient_sync"]["algorithm"] == "hierarchical_ring"
    assert report["checkpoint"]["image"]["sharded"] is True
    assert report["measured"] is False
