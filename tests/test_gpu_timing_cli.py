import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_timing_cli
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_timing_config import small
from test_gpu_timing_sweep import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
LOAD = (1.5, 2.0, 2.5)
# A None entry in sys.modules makes every import of that name fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.gpu_timing_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(tmp_path, payload=None):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload or small()), encoding="utf-8")
    return path


def run_stand_in(tmp_path, output=None):
    config = write_config(tmp_path)
    output = output or tmp_path / "out" / "report.json"
    backend = StandInBackend()
    code = gpu_timing_cli.main(
        ["run", "--config", str(config), "--output", str(output)],
        backend=backend,
        clock=backend.clock,
        load_average=lambda: LOAD,
    )
    return code, config, output


def without_torch(*argv):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *argv],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_a_stand_in_run_prints_synthetic_lines_and_writes_a_report(tmp_path, capsys):
    code, _, output = run_stand_in(tmp_path)

    assert code == gpu_timing_cli.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert (
        lines[0]
        == "synthetic_timing load_average before=1.50,2.00,2.50 after=1.50,2.00,2.50"
    )
    assert lines[1].startswith("synthetic_timing empty_kernel event_median_us=")
    points = [line for line in lines if " point " in line]
    assert len(points) == 3 * 3
    assert all("cross_domain=true" in line for line in points)
    assert sum(" fit side=" in line for line in lines) == 3
    assert lines[-1].startswith("gpu_timing_synthetic_timing=")
    assert "measured=false cuda=false nvidia=false" in lines[-1]
    report = json.loads(output.read_bytes())
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"


def test_verify_accepts_the_written_report_and_checks_the_config(tmp_path, capsys):
    _, config, output = run_stand_in(tmp_path)
    capsys.readouterr()

    code = gpu_timing_cli.main(
        ["verify", "--report", str(output), "--config", str(config)]
    )

    assert code == gpu_timing_cli.EXIT_OK
    line = capsys.readouterr().out.strip()
    assert line.startswith("verified_gpu_timing_synthetic_timing=")
    assert "sweep_points=3 timed_samples=90" in line
    (tmp_path / "other").mkdir()
    other = write_config(tmp_path / "other", small(seed=9))
    code = gpu_timing_cli.main(
        ["verify", "--report", str(output), "--config", str(other)]
    )
    assert code == gpu_timing_cli.EXIT_REJECTED


def test_a_tampered_report_exits_1_and_bad_input_exits_2(tmp_path, capsys):
    _, config, output = run_stand_in(tmp_path)
    report = json.loads(output.read_bytes())
    report["summary"]["sweep_points"] = 7
    output.write_bytes(canonical_json_bytes(report) + b"\n")

    assert gpu_timing_cli.main(["verify", "--report", str(output)]) == 1
    missing = tmp_path / "missing.json"
    assert gpu_timing_cli.main(["verify", "--report", str(missing)]) == 2
    assert "rejected" in capsys.readouterr().err


def test_a_run_refuses_an_output_that_aliases_its_config(tmp_path):
    config = write_config(tmp_path)

    code = gpu_timing_cli.main(
        ["run", "--config", str(config), "--output", str(config)],
        backend=StandInBackend(),
    )

    assert code == gpu_timing_cli.EXIT_INVALID
    assert json.loads(config.read_text(encoding="utf-8")) == small()


def test_an_invalid_config_exits_2(tmp_path):
    config = write_config(tmp_path, small(sweep__repeats=1))

    code = gpu_timing_cli.main(
        ["run", "--config", str(config), "--output", str(tmp_path / "r.json")],
        backend=StandInBackend(),
    )

    assert code == gpu_timing_cli.EXIT_INVALID


def test_a_host_without_torch_exits_3_before_measuring(tmp_path):
    config = write_config(tmp_path)
    output = tmp_path / "report.json"

    completed = without_torch("run", "--config", str(config), "--output", str(output))

    assert completed.returncode == gpu_timing_cli.EXIT_UNAVAILABLE, completed.stderr
    assert "gpu-timing run unavailable:" in completed.stderr
    assert not output.exists()


def test_verify_runs_without_torch(tmp_path, capsys):
    _, config, output = run_stand_in(tmp_path)

    completed = without_torch(
        "verify", "--report", str(output), "--config", str(config)
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("verified_gpu_timing_synthetic_timing=")


def test_the_parser_names_the_program_and_refuses_abbreviations(capsys):
    parser = gpu_timing_cli.build_parser()

    assert parser.prog == gpu_timing_cli.PROG == "ai-journey-gpu-timing"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--conf", "a", "--output", "b"])
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(["--help"])
    assert stopped.value.code == 0
    assert "{run,verify}" in capsys.readouterr().out
