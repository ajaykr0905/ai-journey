import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.memory_access_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    main,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report_io import load_report
from test_memory_access_plan import make_plan_mapping

ROOT = Path(__file__).resolve().parents[1]


def write_plan(path: Path, **changes) -> Path:
    path.write_text(json.dumps({**make_plan_mapping(), **changes}), encoding="utf-8")
    return path


def fake_clock():
    return itertools.count(0, 750).__next__


def test_analyze_seals_and_verify_accepts_the_analytical_report(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "out" / "analysis.json"

    assert main(["analyze", "--config", str(config), "--output", str(output)]) == 0
    built = capsys.readouterr().out
    report = load_report(output)
    assert f"memory_access_analysis={report['report_sha256']}" in built
    assert "patterns=9 transposes=2 tiled_estimates=4" in built
    assert "measured=false gpu_executed=false" in built

    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    verified = capsys.readouterr().out
    assert f"verified_memory_access_analysis={report['report_sha256']}" in verified


def test_benchmark_labels_every_printed_number_as_a_cpu_host_measurement(
    tmp_path, capsys
):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "measurement.json"

    code = main(
        ["benchmark", "--config", str(config), "--output", str(output)],
        clock=fake_clock(),
    )

    assert code == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report = load_report(output)
    assert lines[-1] == (
        f"memory_access_measurement={report['report_sha256']} cases=4 "
        "scope=cpu_wall_clock_this_host gpu_executed=false"
    )
    assert len(lines) == 5
    assert all(line.startswith("cpu_host_measurement backend=") for line in lines[:4])
    assert "median_ns=750" in lines[0]
    assert report["results"][0]["samples_ns"] == [750, 750, 750]
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0


def test_verify_rejects_tampered_or_rebound_reports_with_exit_one(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    other = write_plan(tmp_path / "other.json", name="other-plan")
    output = tmp_path / "analysis.json"
    assert main(["analyze", "--config", str(config), "--output", str(output)]) == 0
    capsys.readouterr()

    assert main(["verify", "--report", str(output), "--config", str(other)]) == (
        EXIT_REJECTED
    )
    assert "supplied plan" in capsys.readouterr().err

    report = load_report(output)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err

    report["summary"]["patterns"] += 1
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "digest" in capsys.readouterr().err


def test_invalid_inputs_and_protected_outputs_exit_two(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    before = config.read_bytes()

    for command in ("analyze", "benchmark"):
        code = main([command, "--config", str(config), "--output", str(config)])
        assert code == EXIT_INVALID
        assert "must not alias" in capsys.readouterr().err
    assert config.read_bytes() == before

    broken = write_plan(tmp_path / "broken.json", schema_version=2)
    code = main(["analyze", "--config", str(broken), "--output", str(tmp_path / "x")])
    assert code == EXIT_INVALID
    assert "schema_version" in capsys.readouterr().err

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"kind": "memory_access_analysis",', encoding="utf-8")
    assert main(["verify", "--report", str(malformed)]) == EXIT_INVALID
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    capsys.readouterr()


def test_usage_errors_are_reported_by_argparse_with_exit_two(capsys):
    with pytest.raises(SystemExit) as error:
        main(["analyze", "--config", "plan.json"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["profile"])
    assert error.value.code == 2
    capsys.readouterr()


def test_module_entry_point_runs_the_cli(tmp_path):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "analysis.json"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_journey.memory_access_cli",
            "analyze",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "memory_access_analysis=" in completed.stdout
    assert load_report(output)["kind"] == "memory_access_analysis"


def test_symlink_loop_paths_exit_two_without_a_traceback(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop)
    output = tmp_path / "out.json"

    for command in ("analyze", "benchmark"):
        code = main(
            [command, "--config", str(config), "--output", str(loop)],
            clock=fake_clock(),
        )
        assert code == EXIT_INVALID
        assert "non-symlink" in capsys.readouterr().err
        code = main([command, "--config", str(loop), "--output", str(output)])
        assert code == EXIT_INVALID
        assert "non-symlink" in capsys.readouterr().err
    assert loop.is_symlink()
    assert not output.exists()
