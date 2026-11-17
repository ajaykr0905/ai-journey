import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_launch_sweep_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.metal_launch_sweep_io import load_report, save_report
from ai_journey.metal_launch_sweep_shapes import MetalLaunchSweepError
from test_kernel_metal_runtime import requires_metal
from test_metal_launch_sweep_config import make_config_mapping
from test_metal_launch_sweep_standin import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
MATMUL = (
    r"matmul m=\d+ n=\d+ k=\d+ group=\d+x\d+ rank=\d median_us=\d+\.\d\d "
    r"iqr_us=\d+\.\d\d gflops_per_second=\d+\.\d relative_to_fastest=\d\.\d{3}"
)
VECTOR = (
    r"vector length=\d+ group=\d+ rank=\d median_us=\d+\.\d\d iqr_us=\d+\.\d\d "
    r"gb_per_second=\d+\.\d\d relative_to_fastest=\d\.\d{3}"
)
RANKING = r"ranking (matmul m=\d+ n=\d+ k=\d+|vector length=\d+) order=[0-9x,]+ "
RANKING += r"spread=\d\.\d{3} separated=(true|false)"
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.metal_launch_sweep_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(path: Path, **changes) -> Path:
    path.write_text(json.dumps({**make_config_mapping(), **changes}), encoding="utf-8")
    return path


def scripted_clock():
    return itertools.count(0, 1_000).__next__


def check_lines(lines, label):
    # 3 pipelines, 3 + 4 admissions, 2 correctness lines, 2 matmul points of
    # 3 shapes, 2 vector points of 4 sizes, a ranking per point, the load line
    # and the summary line.
    assert len(lines) == 3 + 7 + 2 + 2 * 4 + 2 * 5 + 1 + 1
    for line in lines[:3]:
        assert re.fullmatch(
            f"{label} pipeline kernel=[a-z_]+ max_threads_per_threadgroup=\\d+ "
            "thread_execution_width=\\d+ requires_whole_simd_groups=false",
            line,
        ), line
    assert lines[3] == (
        f"{label} admission sweep=matmul group=8x8 threads=64 simd_groups=2 "
        "idle_lanes=0 rows_per_simd_group=4 columns_per_simd_group=8 admitted=true"
    )
    assert lines[10].startswith(f"{label} correctness matmul cases=2 mapping_checks=3 ")
    assert lines[11] == (
        f"{label} correctness vector_add cases=4 "
        "implementations=metal_kernel/mps,torch/mps,torch/cpu"
    )
    body = lines[12:-2]
    for line in body:
        assert re.fullmatch(f"{label} ({MATMUL}|{VECTOR}|{RANKING})", line), line
    assert re.fullmatch(
        f"{label} load_average before=(unknown|[0-9.,]+) after=(unknown|[0-9.,]+)",
        lines[-2],
    ), lines[-2]


def without_torch(*arguments):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def stand_in_run(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "out" / "report.json"
    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, backend=StandInBackend(), clock=scripted_clock()) == EXIT_OK
    return config, output, capsys.readouterr().out.splitlines()


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == "ai-journey-metal-launch-sweep"
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
    config, output, lines = stand_in_run(tmp_path, capsys)

    report, _ = load_report(output)
    check_lines(lines, "synthetic_timing")
    assert (report["measured"], report["backend"]) == (False, "injected")
    assert lines[-1] == (
        f"metal_launch_sweep_synthetic_timing={report['report_sha256']} "
        "admitted_shapes=7 rejected_shapes=0 correctness_cases=6 "
        "benchmark_points=4 scope=synthetic_backend_or_clock measured=false "
        "cuda=false nvidia=false"
    )
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main(argv) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_rejected_shapes_are_printed_with_their_reason(tmp_path, capsys):
    from ai_journey.kernel_metal_geometry import PipelineLimits

    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    backend = StandInBackend(PipelineLimits(128, 32, 0))

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, backend=backend, clock=scripted_clock()) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    rejected = [line for line in lines if "admitted=false" in line]
    assert [line.split(" group=")[1].split()[0] for line in rejected] == [
        "16x16",
        "4x64",
        "256",
    ]
    assert all(
        line.endswith("reason=exceeds_max_threads_per_threadgroup") for line in rejected
    )
    assert "rejected_shapes=3" in lines[-1]


def test_run_stops_with_a_distinct_code_where_torch_cannot_be_imported(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    completed = without_torch("run", "--config", str(config), "--output", str(output))

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    on_macos = sys.platform == "darwin"
    reason = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert completed.stderr.startswith("metal-launch-sweep run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert "needs an Apple GPU on macOS" in completed.stderr
    assert completed.stdout == ""
    assert not output.exists()


def test_verify_works_where_torch_cannot_be_imported(tmp_path, capsys):
    _, output, lines = stand_in_run(tmp_path, capsys)

    completed = without_torch("verify", "--report", str(output))

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == f"verified_{lines[-1]}\n"


def test_verify_rejects_tampered_reformatted_or_rebound_reports(tmp_path, capsys):
    config, output, _ = stand_in_run(tmp_path, capsys)
    other = write_config(tmp_path / "other.json", name="other-config")

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
    assert "digest" in capsys.readouterr().err


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
        assert main(argv, backend=StandInBackend()) == EXIT_INVALID, argv
        assert message in capsys.readouterr().err, argv
    over_budget = write_config(
        tmp_path / "budget.json",
        timing={**make_config_mapping()["timing"], "max_runtime_seconds": 1},
    )
    argv = ["run", "--config", str(over_budget), "--output", str(tmp_path / "b.json")]
    clock = itertools.count(0, 10**8).__next__
    assert main(argv, backend=StandInBackend(), clock=clock) == EXIT_INVALID
    assert "max_runtime_seconds=1" in capsys.readouterr().err
    assert not (tmp_path / "b.json").exists()


def test_a_device_error_during_a_run_exits_two_not_the_rejected_code(tmp_path, capsys):
    # PyTorch reports MPS failures, such as running out of device memory, as
    # RuntimeError. Exit 1 must keep its one meaning: a rejected report.
    class DeviceFails(StandInBackend):
        def matmul_workload(self, a, b, launches):
            raise RuntimeError("MPS backend out of memory")

    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, backend=DeviceFails(), clock=scripted_clock()) == EXIT_INVALID
    assert capsys.readouterr().err == (
        "metal-launch-sweep run failed: MPS backend out of memory\n"
    )
    assert not output.exists()


def test_reports_are_verified_before_they_are_written(tmp_path, capsys):
    _, output, _ = stand_in_run(tmp_path, capsys)
    report, _ = load_report(output)
    report["summary"]["rejected_shapes"] = 5
    target = tmp_path / "copy.json"

    with pytest.raises(MetalLaunchSweepError, match="digest"):
        save_report(target, report)
    assert not target.exists()
    with pytest.raises(MetalLaunchSweepError, match="regular non-symlink"):
        load_report(tmp_path / "absent.json")


@requires_metal
def test_a_run_on_this_gpu_is_a_labeled_measurement_that_verifies(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    lines = capsys.readouterr().out.splitlines()
    check_lines(lines, "apple_gpu_metal_measurement")
    report, _ = load_report(output)
    assert lines[-1] == (
        f"metal_launch_sweep_measurement={report['report_sha256']} "
        "admitted_shapes=7 rejected_shapes=0 correctness_cases=6 "
        "benchmark_points=4 scope=apple_gpu_metal_this_host cuda=false nvidia=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
