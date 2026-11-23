import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.metal_liger_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
)
from ai_journey.metal_liger_io import load_report
from test_metal_fused_ceiling import write_recorded
from test_metal_liger_emulator import LigerEmulator, ticking_clock
from test_metal_liger_report import SMALL_CONFIG

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config" / "day-80-metal-liger.json"
LOAD = (1.5, 2.25, 3.0)


def inputs(directory):
    config = directory / "config.json"
    config.write_text(json.dumps(SMALL_CONFIG), encoding="utf-8")
    return config, write_recorded(directory)


def run_argv(config, ceilings, output):
    return [
        "run",
        "--config",
        str(config),
        "--ceilings",
        str(ceilings),
        "--output",
        str(output),
    ]


def stand_in_main(argv, backend=None):
    return main(
        argv,
        backend=backend or LigerEmulator(),
        clock=ticking_clock(),
        load_average=lambda: LOAD,
    )


def without_torch(*argv):
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_liger_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script, *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_the_parser_names_two_commands_without_abbreviations():
    parser = build_parser()

    assert parser.prog == PROG == "ai-journey-metal-liger"
    for argv in (["run", "--conf", "x"], ["verify"], [], ["plot"]):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)


def test_a_stand_in_run_prints_the_plan_and_every_number(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    cases = len(report["correctness"]["cases"])
    points = len(report["benchmark"]["points"])
    assert len(lines) == 3 + 2 + 3 + cases + points + points + 1 + 1
    assert lines[0].startswith(
        "synthetic_timing plan family=rmsnorm kernel=liger_rmsnorm "
        "base=torch_rmsnorm predicted=faster kernel_sha256="
    )
    assert lines[3] == (
        "synthetic_timing load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    assert lines[5].startswith("synthetic_timing probe function=exp ")
    assert any("correctness family=cross_entropy case=1" in line for line in lines)
    assert any("naive_finite=false" in line for line in lines)
    assert any(
        line.startswith("synthetic_timing speedup family=swiglu") for line in lines
    )
    assert lines[-1].startswith(
        f"metal_liger_synthetic_timing={report['report_sha256']} "
    )
    assert "measured=false cuda=false nvidia=false" in lines[-1]
    assert "liger_code=false backward=false" in lines[-1]


def test_verify_accepts_its_report_and_rejects_a_tampered_one(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    summary = capsys.readouterr().out.splitlines()[-1]

    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main([*argv, "--ceilings", str(ceilings)]) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{summary}\n"

    report = json.loads(output.read_text(encoding="utf-8"))
    report["summary"]["warnings"] = 7
    output.write_text(json.dumps(report), encoding="utf-8")
    assert main(argv) == EXIT_REJECTED
    assert "rejected: report digest" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "none.json")]) == EXIT_INVALID


def test_run_refuses_inputs_and_outputs_before_it_measures(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, ceilings, config)) == EXIT_INVALID
    assert stand_in_main(run_argv(config, ceilings, ceilings)) == EXIT_INVALID
    assert stand_in_main(run_argv(config, ceilings, tmp_path)) == EXIT_INVALID
    missing = tmp_path / "missing.json"
    assert stand_in_main(run_argv(config, missing, output)) == EXIT_INVALID
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({**SMALL_CONFIG, "seed": -1}), encoding="utf-8")
    assert stand_in_main(run_argv(broken, ceilings, output)) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "must not alias the config" in errors
    assert "must not alias the ceiling report" in errors
    assert "ceiling report was refused" in errors
    assert not output.exists()


def test_a_check_that_fails_on_the_device_writes_no_report(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"

    def corrupt(family, buffer):
        buffer = buffer.copy()
        buffer[-1] = 0.0
        return buffer

    backend = LigerEmulator(corrupt=corrupt)
    assert stand_in_main(run_argv(config, ceilings, output), backend) == EXIT_INVALID
    assert "escaped into a gap slot" in capsys.readouterr().err
    assert not output.exists()


def test_a_host_without_torch_exits_3_and_writes_nothing(tmp_path):
    _, ceilings = inputs(tmp_path)
    output = tmp_path / "new" / "report.json"

    completed = without_torch(*run_argv(PINNED, ceilings, output))

    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("metal-liger run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_runs_without_torch(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    summary = capsys.readouterr().out.splitlines()[-1]

    completed = without_torch(
        "verify", "--report", str(output), "--ceilings", str(ceilings)
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == f"verified_{summary}\n"
