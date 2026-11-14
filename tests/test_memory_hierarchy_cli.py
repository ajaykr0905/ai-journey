import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_hierarchy_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    build_parser,
    main,
)
from ai_journey.memory_hierarchy_io import load_report
from test_memory_hierarchy_config import make_config_mapping

ROOT = Path(__file__).resolve().parents[1]
LINE = {
    "stream": r"stream kernel=(copy|scale|add|triad) points=3 "
    r"peak_gb_per_second=\d+\.\d{3} peak_working_set_bytes=\d+ "
    r"largest_gb_per_second=\d+\.\d{3} largest_working_set_bytes=\d+",
    "access": r"access pattern=(copy|sequential|strided|random) "
    r"median_ns_per_element=\d+\.\d{3} median_gb_per_second=\d+\.\d{3} iqr_ns=\d+",
    "matmul": r"matmul m=32 n=24 k=16 dtype=float64 median_gflops_per_second=\d+\.\d{3}",
}


def write_config(path: Path, **changes) -> Path:
    path.write_text(json.dumps({**make_config_mapping(), **changes}), encoding="utf-8")
    return path


def scripted_clock():
    return itertools.count(0, 1_000).__next__


def check_lines(lines, label):
    assert len(lines) == 12
    for line in lines[:4]:
        assert re.fullmatch(f"{label} {LINE['stream']}", line), line
    for line in lines[4:8]:
        assert re.fullmatch(f"{label} {LINE['access']}", line), line
    assert re.fullmatch(f"{label} {LINE['matmul']}", lines[8]), lines[8]
    assert re.fullmatch(
        r"derived roofline ridge_point_flops_per_byte=\d+\.\d{3} "
        r"compute_ceiling_gflops_per_second=\d+\.\d{3} "
        r"bandwidth_ceiling_gb_per_second=\d+\.\d{3}",
        lines[9],
    ), lines[9]
    assert re.fullmatch(
        r"derived littles_law stated_latency_ns=100 triad_bytes_in_flight=\d+\.\d "
        r"random_elements_in_flight=\d+\.\d{2} latency_measured=false",
        lines[10],
    ), lines[10]


def test_run_measures_this_host_and_verify_accepts_the_report(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "out" / "report.json"

    assert main(["run", "--config", str(config), "--output", str(output)]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)
    check_lines(lines, "cpu_host_measurement")
    assert lines[-1] == (
        f"memory_hierarchy_measurement={report['report_sha256']} "
        "stream_measurements=12 access_patterns=4 scope=cpu_wall_clock_this_host "
        "gpu_executed=false"
    )
    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main(argv) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_an_injected_clock_run_prints_and_verifies_as_synthetic(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "synthetic.json"

    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, clock=scripted_clock()) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)
    check_lines(lines, "synthetic_timing")
    assert (report["timer"], report["measured"]) == ("injected", False)
    assert lines[-1] == (
        f"memory_hierarchy_synthetic_timing={report['report_sha256']} "
        "stream_measurements=12 access_patterns=4 scope=synthetic_injected_clock "
        "measured=false gpu_executed=false"
    )
    assert main(["verify", "--report", str(output)]) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_verify_rejects_tampered_reformatted_or_rebound_reports(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    other = write_config(tmp_path / "other.json", name="other-config")
    output = tmp_path / "report.json"
    argv = ["run", "--config", str(config), "--output", str(output)]
    assert main(argv, clock=scripted_clock()) == EXIT_OK
    capsys.readouterr()

    rebound = ["verify", "--report", str(output), "--config", str(other)]
    assert main(rebound) == EXIT_REJECTED
    assert "supplied config" in capsys.readouterr().err
    report, _ = load_report(output)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    report["roofline"]["ridge_point_flops_per_byte"] *= 2
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "digest" in capsys.readouterr().err


def test_invalid_inputs_refused_outputs_and_failed_runs_exit_two(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()

    assert main(["run", "--config", str(config), "--output", str(config)]) == (
        EXIT_INVALID
    )
    assert "must not alias" in capsys.readouterr().err
    assert config.read_bytes() == before
    broken = write_config(tmp_path / "broken.json", schema_version=2)
    output = tmp_path / "report.json"
    assert main(["run", "--config", str(broken), "--output", str(output)]) == (
        EXIT_INVALID
    )
    assert "schema_version" in capsys.readouterr().err
    timing = {**make_config_mapping()["timing"], "max_runtime_seconds": 1}
    slow = write_config(tmp_path / "slow.json", timing=timing)
    argv = ["run", "--config", str(slow), "--output", str(output)]
    assert main(argv, clock=itertools.count(0, 10**8).__next__) == EXIT_INVALID
    assert "max_runtime_seconds=1" in capsys.readouterr().err
    assert not output.exists()
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"kind": ', encoding="utf-8")
    assert main(["verify", "--report", str(malformed)]) == EXIT_INVALID
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == (
        EXIT_INVALID
    )
    capsys.readouterr()


def test_unusable_outputs_are_refused_before_anything_is_measured(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    directory = tmp_path / "directory"
    directory.mkdir()
    target = tmp_path / "target.json"
    target.write_bytes(b"keep\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    def unused_clock():
        raise AssertionError("the run started measuring")

    for output in (directory, link):
        argv = ["run", "--config", str(config), "--output", str(output)]
        assert main(argv, clock=unused_clock) == EXIT_INVALID
        assert "regular non-symlink file" in capsys.readouterr().err
    assert target.read_bytes() == b"keep\n"


def test_outputs_the_writer_cannot_create_are_refused_before_measuring(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    blocker = tmp_path / "blocker.json"
    blocker.write_bytes(b"keep\n")
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    locked = tmp_path / "locked"
    locked.mkdir()
    outputs = [
        blocker / "report.json",
        blocker / "nested" / "report.json",
        loop / "report.json",
    ]
    # Root can write into a read-only directory, so only users check it.
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        outputs.append(locked / "report.json")

    def unused_clock():
        raise AssertionError("the run started measuring")

    locked.chmod(0o500)
    try:
        for output in outputs:
            argv = ["run", "--config", str(config), "--output", str(output)]
            assert main(argv, clock=unused_clock) == EXIT_INVALID
            assert "writable directory" in capsys.readouterr().err
    finally:
        locked.chmod(0o700)
    assert blocker.read_bytes() == b"keep\n"
    assert not list(locked.iterdir())


def test_symlink_loops_exit_two_without_a_traceback(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    loop = tmp_path / "loop.json"
    loop.symlink_to(loop)
    output = tmp_path / "report.json"

    argv = ["run", "--config", str(config), "--output", str(loop)]
    assert main(argv, clock=scripted_clock()) == EXIT_INVALID
    assert "non-symlink" in capsys.readouterr().err
    assert main(["run", "--config", str(loop), "--output", str(output)]) == (
        EXIT_INVALID
    )
    assert "non-symlink" in capsys.readouterr().err
    assert loop.is_symlink() and not output.exists()


def test_usage_errors_exit_two_and_help_names_the_console_script(capsys):
    for argv in (["run", "--config", "config.json"], ["profile"], ["verify"]):
        with pytest.raises(SystemExit) as stopped:
            main(argv)
        assert stopped.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: ai-journey-memory-hierarchy ")
    assert "{run,verify}" in usage
    assert "no GPU or CUDA kernel" in usage


def test_module_entry_point_runs_the_cli(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
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
            "ai_journey.memory_hierarchy_cli",
            "run",
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
    assert "memory_hierarchy_measurement=" in completed.stdout
    assert load_report(output)[0]["measured"] is True
