import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_ceiling_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.gpu_ceiling_io import load_report
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_backend import NumpyBackend
from test_gpu_ceiling_config import make_config_mapping
from test_gpu_ceiling_report import loads, varied_clock
from test_kernel_metal_runtime import requires_metal

ROOT = Path(__file__).resolve().parents[1]
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.gpu_ceiling_cli import main
sys.exit(main(sys.argv[1:]))
"""
NUMBER = r"-?\d+\.\d+"
LINES = [
    rf"host_load before={NUMBER},{NUMBER},{NUMBER} after={NUMBER},{NUMBER},{NUMBER}",
    r"correctness fma_cases=2 stream_cases=2 bank_cases=6 exact=true",
    *[
        rf"fma chains={chains} fma_steps={steps} median_us={NUMBER} "
        rf"iqr_us={NUMBER} gflops_per_second={NUMBER}"
        for chains in (1, 4)
        for steps in (8, 32, 128)
    ],
    *[
        rf"stream kernel={kernel} buffer_bytes={size} calls_per_sample=\d+ "
        rf"median_us={NUMBER} gb_per_second={NUMBER}"
        for kernel in ("stream_copy", "stream_read_reduce")
        for size in (16384, 65536)
    ],
    *[
        rf"bank row_words={words} stride={stride} ps_per_access={NUMBER} "
        rf"relative_to_unpadded_stride_1={NUMBER}"
        for words in (32, 33)
        for stride in (1, 2, 32)
    ],
    rf"ceiling compute gflops_per_second={NUMBER} chains=(1|4) "
    rf"fma_steps=(8|32|128) fixed_share={NUMBER} scales_with_work=(true|false)",
    rf"ceiling bandwidth gb_per_second={NUMBER} "
    r"kernel=stream_(copy|read_reduce) buffer_bytes=65536",
    rf"ceiling ridge_flops_per_byte={NUMBER}",
    r"bank finding=(no_clear_conflict_pattern|stride_slowdown_removed_by_padding|"
    rf"stride_slowdown_reduced_by_padding|no_single_pattern) "
    rf"largest_unpadded_ratio={NUMBER} largest_padded_ratio={NUMBER}",
]


def write_config(path: Path, **changes) -> Path:
    path.write_text(json.dumps({**make_config_mapping(), **changes}), encoding="utf-8")
    return path


def stand_in(**changes):
    return {
        "backend": NumpyBackend(),
        "clock": varied_clock(),
        "load_average": loads(),
        **changes,
    }


def check_lines(lines, label):
    assert len(lines) == len(LINES) + 1
    for line, pattern in zip(lines, LINES):
        assert re.fullmatch(f"{label} {pattern}", line), line


def without_torch(*arguments):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == "ai-journey-gpu-ceilings"
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
    output = tmp_path / "out" / "report.json"

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, **stand_in()) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)
    check_lines(lines, "synthetic_timing")
    assert (
        lines[0]
        == "synthetic_timing host_load before=3.50,4.25,5.00 after=3.75,4.50,5.00"
    )
    assert lines[-1] == (
        f"gpu_ceiling_synthetic_timing={report['report_sha256']} "
        "correctness_cases=10 fma_points=6 stream_points=4 bank_points=6 "
        "scope=synthetic_backend_or_clock measured=false vendor_peak=false "
        "cuda=false nvidia=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_verify_rejects_rebound_reformatted_or_tampered_reports(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    other = write_config(tmp_path / "other.json", name="other-config")
    output = tmp_path / "report.json"
    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, **stand_in()) == EXIT_OK
    capsys.readouterr()

    rebound = ["verify", "--report", str(output), "--config", str(other)]
    assert main(rebound) == EXIT_REJECTED
    assert "supplied config" in capsys.readouterr().err
    report, _ = load_report(output)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    report["ceilings"]["fma"]["gflops_per_second"] *= 2
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert capsys.readouterr().err.startswith("gpu-ceiling verify rejected: ")


def test_run_stops_with_a_distinct_code_where_torch_cannot_be_imported(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    completed = without_torch("run", "--config", str(config), "--output", str(output))

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    on_macos = sys.platform == "darwin"
    reason = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert completed.stderr.startswith("gpu-ceiling run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert "needs an Apple GPU on macOS" in completed.stderr
    assert completed.stdout == ""
    assert not output.exists()


def test_verify_works_where_torch_cannot_be_imported(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, **stand_in()) == EXIT_OK
    capsys.readouterr()

    completed = without_torch(
        "verify", "--report", str(output), "--config", str(config)
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith("verified_gpu_ceiling_synthetic_timing=")


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
        assert main(argv, **stand_in()) == EXIT_INVALID, argv
        assert message in capsys.readouterr().err, argv
    over_budget = write_config(
        tmp_path / "budget.json",
        timing={**make_config_mapping()["timing"], "max_runtime_seconds": 1},
    )
    argv = ["run", "--config", str(over_budget), "--output", str(tmp_path / "b.json")]
    clock = itertools.count(0, 10**8).__next__
    assert main(argv, **stand_in(clock=clock)) == EXIT_INVALID
    assert "max_runtime_seconds=1" in capsys.readouterr().err
    assert not (tmp_path / "b.json").exists()


def test_a_device_error_during_a_run_exits_two_not_the_rejected_code(tmp_path, capsys):
    # PyTorch reports MPS failures, such as running out of device memory, as
    # RuntimeError. Exit 1 must keep its one meaning: a rejected report.
    class DeviceFails(NumpyBackend):
        def stream_workload(self, name, source, launch, **arguments):
            raise RuntimeError("MPS backend out of memory")

    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, **stand_in(backend=DeviceFails())) == EXIT_INVALID
    assert capsys.readouterr().err == (
        "gpu-ceiling run failed: MPS backend out of memory\n"
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
        f"gpu_ceiling_measurement={report['report_sha256']} correctness_cases=10 "
        "fma_points=6 stream_points=4 bank_points=6 scope=apple_gpu_metal_this_host "
        "vendor_peak=false cuda=false nvidia=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
