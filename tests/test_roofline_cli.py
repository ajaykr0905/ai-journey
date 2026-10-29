import json
from pathlib import Path

import pytest

from ai_journey.resource_accounting import ResourceAccountingError
from ai_journey.roofline_cli import load_roofline_plan, main
from ai_journey.roofline_report_io import (
    MAX_ROOFLINE_REPORT_BYTES,
    load_roofline_report,
)
from test_roofline_report import make_plan


def write_plan(path: Path) -> None:
    plan = make_plan()
    path.write_text(
        json.dumps(
            {
                "model": plan.model.__dict__,
                "batch_size": plan.batch_size,
                "sequence_length": plan.sequence_length,
                "dtype": plan.dtype,
                "peak_flops_per_second": plan.peak_flops_per_second,
                "memory_bandwidth_bytes_per_second": (
                    plan.memory_bandwidth_bytes_per_second
                ),
            }
        ),
        encoding="utf-8",
    )


def test_roofline_cli_builds_and_verifies_sealed_report(tmp_path: Path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    write_plan(config)

    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    built = capsys.readouterr().out
    report = load_roofline_report(output)
    assert report["summary"]["operation_count"] == 10
    assert report["integrity"]["digest"] in built
    assert "measured=false" in built

    assert main(["verify", "--report", str(output)]) == 0
    verified = capsys.readouterr().out
    assert report["integrity"]["digest"] in verified


def test_roofline_cli_rejects_duplicate_oversize_and_invalid_plans(tmp_path: Path):
    config = tmp_path / "plan.json"
    config.write_text('{"model":{},"model":{}}', encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="duplicate"):
        load_roofline_plan(config)

    config.write_bytes(b"{" + b" " * MAX_ROOFLINE_REPORT_BYTES)
    with pytest.raises(ResourceAccountingError, match="exceeds"):
        load_roofline_plan(config)

    config.write_text('{"model":{}}', encoding="utf-8")
    with pytest.raises(ResourceAccountingError, match="unable to load"):
        load_roofline_plan(config)


def test_roofline_cli_preserves_config_when_output_aliases_it(tmp_path: Path):
    config = tmp_path / "plan.json"
    write_plan(config)
    before = config.read_bytes()

    with pytest.raises(ResourceAccountingError, match="must not alias"):
        main(["build", "--config", str(config), "--output", str(config)])

    assert config.read_bytes() == before
