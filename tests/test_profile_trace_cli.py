import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import profile_trace_report
from ai_journey.matmul_metal_runtime import MatmulMetalUnavailableError
from ai_journey.profile_trace_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
)
from ai_journey.profile_trace_io import REPORT_FILE
from test_profile_trace_capture import FakeSession
from test_profile_trace_config import make_config_mapping
from test_profile_trace_events import TORCH_LAYOUT

ROOT = Path(__file__).resolve().parents[1]
LOADS = (1.5, 2.25, 3.0)


def write_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(make_config_mapping()))
    return path


def capture(tmp_path, capsys):
    config = write_config(tmp_path)
    output = tmp_path / "capture"
    code = main(
        ["capture", "--config", str(config), "--output-dir", str(output)],
        session=FakeSession(),
        load_reader=lambda: LOADS,
    )
    return code, config, output, capsys.readouterr()


def test_capture_prints_synthetic_lines_and_publishes_a_verified_directory(
    tmp_path, capsys
):
    code, config, output, printed = capture(tmp_path, capsys)

    assert code == EXIT_OK
    lines = printed.out.splitlines()
    assert (
        lines[0] == "synthetic_timing load_before=1.5,2.25,3.0 load_after=1.5,2.25,3.0"
    )
    assert re.fullmatch(
        r"synthetic_timing shape m=64 n=48 k=32 bottleneck=matmul_x_rows/synchronize "
        r"kind=gpu_wait share=0\.\d{3}",
        lines[1],
    )
    assert re.fullmatch(
        r"synthetic_timing side m=64 x_columns host_us=\d+\.\d dispatch_us=\d+\.\d "
        r"synchronize_us=\d+\.\d device_event_us=\d+\.\d host_gflops=\d+\.\d "
        r"fma_fraction=\d\.\d{4}",
        lines[2],
    )
    assert lines[-2] == f"report {output / REPORT_FILE}"
    assert re.fullmatch(
        r"profile_trace_synthetic_timing=[0-9a-f]{64} shapes=2 scope=synthetic_session "
        r"bottlenecks=64x48x32:matmul_x_rows/synchronize,"
        r"33x17x9:matmul_x_rows/synchronize gpu_kernels_recorded=false cuda=false "
        r"nvidia=false",
        lines[-1],
    )
    digest = lines[-1].split("=")[1].split()[0]
    code = main(
        ["verify", "--report", str(output / REPORT_FILE), "--config", str(config)]
    )
    assert code == EXIT_OK
    assert capsys.readouterr().out.startswith(
        f"verified_profile_trace_synthetic_timing={digest} shapes=2"
    )


def test_verify_exit_codes_separate_rejected_from_invalid(tmp_path, capsys):
    _, config, output, _ = capture(tmp_path, capsys)
    report_path = output / REPORT_FILE
    report = json.loads(report_path.read_text())
    report["label"] = "x"
    report_path.write_text(json.dumps(report))
    assert main(["verify", "--report", str(report_path)]) == EXIT_REJECTED
    assert "rejected: report digest" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "none.json")]) == EXIT_INVALID
    other = tmp_path / "other.json"
    other.write_text(json.dumps(make_config_mapping(seed=1)))
    capture_dir = tmp_path / "second"
    main(
        ["capture", "--config", str(config), "--output-dir", str(capture_dir)],
        session=FakeSession(),
        load_reader=lambda: LOADS,
    )
    capsys.readouterr()
    code = main(
        ["verify", "--report", str(capture_dir / REPORT_FILE), "--config", str(other)]
    )
    assert code == EXIT_REJECTED
    assert "given config" in capsys.readouterr().err


def test_capture_refuses_bad_outputs_and_reports_missing_mps(
    tmp_path, capsys, monkeypatch
):
    config = write_config(tmp_path)
    as_file = tmp_path / "file"
    as_file.write_text("")
    argv = ["capture", "--config", str(config), "--output-dir"]
    assert main(argv + [str(as_file)], session=FakeSession()) == EXIT_INVALID
    assert "real directory" in capsys.readouterr().err
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    aliased = config_dir / REPORT_FILE
    aliased.write_text(config.read_text())
    code = main(
        ["capture", "--config", str(aliased), "--output-dir", str(config_dir)],
        session=FakeSession(),
    )
    assert code == EXIT_INVALID
    assert "alias the config" in capsys.readouterr().err

    def unavailable():
        raise MatmulMetalUnavailableError("no MPS device (mps_unavailable)")

    monkeypatch.setattr(profile_trace_report, "TorchProfilerSession", unavailable)
    assert main(argv + [str(tmp_path / "out")]) == EXIT_UNAVAILABLE
    assert "unavailable: no MPS device" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_analyze_reads_any_torch_export(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_bytes(TORCH_LAYOUT)
    code = main(["analyze", "--trace", str(path), "--top", "2", "--width", "16"])

    assert code == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "trace events=6 spans=3 ops=2"
    assert lines[1] == "top 1 name=iter_0 calls=1 self_us=7317.3 total_us=7532.1"
    assert lines[2] == "top 2 name=aten::mm calls=1 self_us=214.9 total_us=214.9"
    assert lines[3].startswith("gaps pid=33879 tid=43692759 calls=1")
    assert lines[4].startswith("timeline extent_ns=7532139 columns=16")
    path.write_bytes(b'{"traceEvents": [{"ph": "C"}]}')
    assert main(["analyze", "--trace", str(path)]) == EXIT_INVALID
    assert "phase 'C'" in capsys.readouterr().err


def test_analyze_reports_iterations_of_a_capture_trace(tmp_path, capsys):
    _, _, output, _ = capture(tmp_path, capsys)
    trace = output / "trace-0-64x48x32.json"
    assert main(["analyze", "--trace", str(trace)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "top 1 name=matmul_x_rows/synchronize calls=3" in out
    assert re.search(r"iterations name=profile_trace/iteration count=3 ", out)


def test_the_parser_has_no_abbreviations_and_the_module_runs():
    parser = build_parser()
    assert parser.prog == PROG
    with pytest.raises(SystemExit):
        parser.parse_args(["capture", "--conf", "x", "--output-dir", "y"])
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "--trace", "x", "--depth", "9"])
    completed = subprocess.run(
        [sys.executable, "-m", "ai_journey.profile_trace_cli", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
    )
    assert "no CUDA kernel, NVIDIA GPU or Nsight tool" in " ".join(
        completed.stdout.split()
    )
