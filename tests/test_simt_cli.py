import os
from pathlib import Path
import subprocess
import sys

import pytest

from ai_journey.simt_cli import EXIT_INVALID, EXIT_OK, main
from ai_journey.simt_json import canonical_json_bytes
from ai_journey.simt_report_io import load_simt_report
from test_simt_plan import plan_mapping, write_plan

ROOT = Path(__file__).resolve().parents[1]


def test_build_then_verify_print_the_same_digest(tmp_path: Path, capsys):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "out" / "report.json"

    assert main(["build", "--config", str(config), "--output", str(output)]) == EXIT_OK
    built = capsys.readouterr().out.strip()
    digest = load_simt_report(output)["integrity"]["digest"]
    assert built == (
        f"simt_report={digest} devices=2 sweep_cells=8 memory_rows=2 "
        "arithmetic_rows=4 measured=false gpu_executed=false"
    )

    assert main(["verify", "--report", str(output)]) == EXIT_OK
    verified = capsys.readouterr().out.strip()
    assert verified == built.replace("simt_report=", "verified_simt_report=", 1)


def test_build_refuses_to_overwrite_its_own_plan(tmp_path: Path, capsys):
    config = write_plan(tmp_path / "plan.json")
    before = config.read_bytes()
    link = tmp_path / "link.json"
    os.link(config, link)

    for target in (config, link):
        code = main(["build", "--config", str(config), "--output", str(target)])
        assert code == EXIT_INVALID
        assert "must not alias a protected input" in capsys.readouterr().err
    assert config.read_bytes() == before


def test_invalid_plans_and_reports_return_the_invalid_exit_code(tmp_path: Path, capsys):
    config = tmp_path / "plan.json"
    config.write_bytes(b'{"name":"a","name":"b"}')
    output = tmp_path / "report.json"
    assert main(["build", "--config", str(config), "--output", str(output)]) == 1
    assert "duplicate JSON field: name" in capsys.readouterr().err
    assert not output.exists()

    mapping = plan_mapping()
    mapping["vector_add"]["registers_per_thread"] = 255
    mapping["vector_add"]["threads_per_block"] = 1024
    write_plan(config, mapping)
    assert main(["build", "--config", str(config), "--output", str(output)]) == 1
    assert "not resident" in capsys.readouterr().err

    write_plan(config)
    assert main(["build", "--config", str(config), "--output", str(output)]) == 0
    capsys.readouterr()
    report = load_simt_report(output)
    report["summary"]["devices"] = 3
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_INVALID
    assert "integrity digest" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == 1
    assert "unable to read SIMT report" in capsys.readouterr().err


def test_usage_errors_exit_with_argparse_status():
    with pytest.raises(SystemExit) as error:
        main(["build", "--config", "plan.json"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["verify", "--rep", "report.json"])
    assert error.value.code == 2


def test_module_entry_point_verifies_a_report(tmp_path: Path):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "report.json"
    assert main(["build", "--config", str(config), "--output", str(output)]) == 0

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_journey.simt_cli",
            "verify",
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("verified_simt_report=")
    assert completed.stdout.strip().endswith("measured=false gpu_executed=false")
