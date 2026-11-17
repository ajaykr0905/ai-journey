import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_metal_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.matmul_metal_io import load_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_config import make_config_mapping
from test_matmul_metal_recorded import RECORDED_REPORT_JSON, RecordedBackend

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"\d+(?:\.\d+)?(?:e[-+]\d+)?"
CASE = (
    rf"correctness m=\d+ n=\d+ k=\d+ naive_error={NUMBER} torch_mps_error={NUMBER} "
    rf"torch_cpu_error={NUMBER} bound={NUMBER} naive_same_bits=true"
)
SIDE = r"{0}_median_us=\d+\.\d {0}_iqr_us=\d+\.\d {0}_gflops=\d+\.\d"
POINT = (
    r"benchmark m=\d+ n=\d+ k=\d+ "
    + " ".join(SIDE.format(short) for short in ("x_columns", "x_rows", "torch"))
    + r" x_columns_to_torch=\d+\.\d{3} x_rows_to_torch=\d+\.\d{3}"
)
CEILING = (
    r"ceiling torch_gflops=\d+\.\d at m=\d+ n=\d+ k=\d+ "
    r"x_columns_best_gflops=\d+\.\d x_columns_fraction=\d+\.\d{3} "
    r"x_rows_best_gflops=\d+\.\d x_rows_fraction=\d+\.\d{3}"
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_metal_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(path: Path, **changes) -> Path:
    path.write_text(json.dumps({**make_config_mapping(), **changes}), encoding="utf-8")
    return path


def scripted_clock():
    return itertools.count(0, 1_000).__next__


def check_lines(lines, label):
    assert len(lines) == 3 + 2 + 1 + 1
    for line in lines[:3]:
        assert re.fullmatch(f"{label} {CASE}", line), line
    for line in lines[3:5]:
        assert re.fullmatch(f"{label} {POINT}", line), line
    assert re.fullmatch(f"{label} {CEILING}", lines[5]), lines[5]


def without_torch(*arguments):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def stand_in_run(tmp_path, config):
    output = tmp_path / "report.json"
    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, backend=RecordedBackend(), clock=scripted_clock()) == EXIT_OK
    return output


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == "ai-journey-matmul-metal"
    parsed = parser.parse_args(["run", "--config", "c.json", "--output", "r.json"])
    assert (parsed.command, str(parsed.config), str(parsed.output)) == (
        "run",
        "c.json",
        "r.json",
    )
    assert parser.parse_args(["verify", "--report", "r.json"]).config is None
    for argv in ([], ["run", "--config", "c.json"], ["run", "--conf", "c.json"]):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)


def test_a_stand_in_run_prints_and_verifies_as_synthetic(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")

    output = stand_in_run(tmp_path, config)
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    check_lines(lines, "synthetic_timing")
    assert (report["measured"], report["backend"]) == (False, "injected")
    assert lines[-1] == (
        f"matmul_metal_synthetic_timing={report['report_sha256']} "
        "correctness_cases=3 benchmark_points=2 scope=synthetic_backend_or_clock "
        "measured=false cuda=false nvidia=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_the_recorded_measurement_verifies_with_its_labels(tmp_path, capsys):
    path = tmp_path / "recorded.json"
    path.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")
    report = json.loads(RECORDED_REPORT_JSON)

    assert main(["verify", "--report", str(path)]) == EXIT_OK
    assert capsys.readouterr().out == (
        f"verified_matmul_metal_measurement={report['report_sha256']} "
        "correctness_cases=3 benchmark_points=2 scope=apple_gpu_metal_this_host "
        "cuda=false nvidia=false\n"
    )
    completed = without_torch("verify", "--report", str(path))
    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith("verified_matmul_metal_measurement=")


def test_run_stops_with_a_distinct_code_where_torch_cannot_be_imported(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    completed = without_torch("run", "--config", str(config), "--output", str(output))

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    on_macos = sys.platform == "darwin"
    reason = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert completed.stderr.startswith("matmul-metal run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert "needs an Apple GPU on macOS" in completed.stderr
    assert completed.stdout == ""
    assert not output.exists()


def test_verify_works_where_torch_cannot_be_imported(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = stand_in_run(tmp_path, config)
    capsys.readouterr()

    completed = without_torch(
        "verify", "--report", str(output), "--config", str(config)
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith("verified_matmul_metal_synthetic_timing=")


def test_verify_rejects_tampered_reformatted_or_rebound_reports(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    other = write_config(tmp_path / "other.json", name="other-config")
    output = stand_in_run(tmp_path, config)
    capsys.readouterr()

    rebound = ["verify", "--report", str(output), "--config", str(other)]
    assert main(rebound) == EXIT_REJECTED
    assert "supplied config" in capsys.readouterr().err
    report, _ = load_report(output)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    report["summary"]["benchmark_points"] += 1
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "matmul-metal verify rejected: " in capsys.readouterr().err


def test_invalid_inputs_refused_outputs_and_failed_runs_exit_two(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    invalid = write_config(tmp_path / "invalid.json", seed=-1)
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "elsewhere.json")
    blocked = tmp_path / "file.txt"
    blocked.write_text("x", encoding="utf-8")
    cases = [
        (["run", "--config", str(tmp_path / "missing.json")], "regular non-symlink"),
        (["run", "--config", str(invalid)], "seed must be in"),
        (["run", "--config", str(config), "--output", str(config)], "alias"),
        (["run", "--config", str(config), "--output", str(tmp_path)], "regular"),
        (["run", "--config", str(config), "--output", str(link)], "regular"),
        (
            ["run", "--config", str(config), "--output", str(blocked / "r.json")],
            "writable directory",
        ),
        (["verify", "--report", str(tmp_path / "missing.json")], "regular"),
    ]
    for argv, message in cases:
        if argv[0] == "run" and "--output" not in argv:
            argv = [*argv, "--output", str(tmp_path / "out.json")]
        assert main(argv, backend=RecordedBackend()) == EXIT_INVALID, argv
        assert message in capsys.readouterr().err, argv
    over_budget = write_config(
        tmp_path / "budget.json",
        benchmark={**make_config_mapping()["benchmark"], "max_runtime_seconds": 1},
    )
    argv = ["run", "--config", str(over_budget), "--output", str(tmp_path / "b.json")]
    clock = itertools.count(0, 10**8).__next__
    assert main(argv, backend=RecordedBackend(), clock=clock) == EXIT_INVALID
    assert "max_runtime_seconds=1" in capsys.readouterr().err
    assert not (tmp_path / "b.json").exists()


def test_a_device_error_during_a_run_exits_two_not_the_rejected_code(tmp_path, capsys):
    # PyTorch reports MPS failures, such as running out of device memory, as
    # RuntimeError. Exit 1 must keep its one meaning: a rejected report.
    class DeviceFails(RecordedBackend):
        def run(self, name, left, right, launch):
            raise RuntimeError("MPS backend out of memory")

    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, backend=DeviceFails(), clock=scripted_clock()) == EXIT_INVALID
    assert capsys.readouterr().err == (
        "matmul-metal run failed: MPS backend out of memory\n"
    )
    assert not output.exists()


@requires_metal
def test_a_run_on_this_gpu_is_a_labeled_measurement_that_verifies(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    lines = capsys.readouterr().out.splitlines()
    check_lines(lines, "apple_gpu_metal_measurement")
    report, _ = load_report(output)
    assert lines[-1] == (
        f"matmul_metal_measurement={report['report_sha256']} correctness_cases=3 "
        "benchmark_points=2 scope=apple_gpu_metal_this_host cuda=false nvidia=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
