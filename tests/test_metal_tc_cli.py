import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The report records whether torch was imported before its warning capture.
# Import it here, so the answer is the same on macOS and on a Linux CI job.
import torch  # noqa: F401

from ai_journey import metal_tc_cli, metal_tc_runtime
from ai_journey.kernel_metal_runtime import probe_metal
from ai_journey.metal_tc_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.metal_tc_io import load_report, save_report
from ai_journey.metal_tc_source import MetalTcError
from test_matmul_roofline_ci_contract import ON_LINUX
from test_metal_fused_ceiling import write_recorded
from test_metal_liger_emulator import ticking_clock
from test_metal_tc_emulator import TcEmulator
from test_metal_tc_report import small_config, synthetic_report

ROOT = Path(__file__).resolve().parents[1]


def inputs(directory):
    config = directory / "config.json"
    config.write_text(json.dumps(small_config().to_mapping()), encoding="utf-8")
    return config, write_recorded(directory)


def run(directory, output, **keywords):
    config, ceilings = inputs(directory)
    argv = ["run", "--config", str(config), "--ceilings", str(ceilings)]
    return main(
        [*argv, "--output", str(output)],
        backend=keywords.get("backend", TcEmulator()),
        clock=ticking_clock(),
        load_average=lambda: (1.0, 2.0, 3.0),
    )


def test_a_run_with_a_stand_in_publishes_a_synthetic_report(tmp_path, capsys):
    output = tmp_path / "report.json"

    assert run(tmp_path, output) == EXIT_OK
    printed = capsys.readouterr().out
    assert "synthetic_timing plan step=m1_over_s1 first=m1 base=s1" in printed
    assert "synthetic_timing support metal_language_version=400 m3=true" in printed
    assert "synthetic_timing benchmark point=1 size=40" in printed
    assert "synthetic_timing verdicts point=0 size=32 m1_over_s1=" in printed
    assert "metal_tc_synthetic_timing=" in printed
    assert "measured=false tensor_cores=false cuda=false nvidia=false" in printed
    config, ceilings = inputs(tmp_path)
    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main([*argv, "--ceilings", str(ceilings)]) == EXIT_OK
    assert "verified_metal_tc_synthetic_timing=" in capsys.readouterr().out


def test_an_unsupported_bfloat_path_prints_its_reason(tmp_path, capsys):
    backend = TcEmulator(bfloat=False)

    assert run(tmp_path, tmp_path / "report.json", backend=backend) == EXIT_OK
    printed = capsys.readouterr().out
    assert "m3=false:the_bfloat_library_did_not_compile" in printed
    assert "m3_over_s1=not_run" in printed
    assert "bfloat_kernel=false" in printed


def test_a_host_without_mps_exits_3_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def linux(*, system, import_module):
        return probe_metal(system=lambda: "Linux", import_module=import_module)

    monkeypatch.setattr(metal_tc_runtime, "probe_metal", linux)
    output = tmp_path / "out" / "report.json"

    assert run(tmp_path, output, backend=None) == EXIT_UNAVAILABLE
    assert not output.parent.exists()
    assert "metal-tc run unavailable: Metal kernels run only on macOS" in (
        capsys.readouterr().err
    )


def test_the_run_on_a_linux_host_stops_before_torch_and_writes_nothing(tmp_path):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "out" / "report.json"
    argv = ["run", "--config", str(config), "--ceilings", str(ceilings)]

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, metal_tc_cli.__name__, *argv]
        + ["--output", str(output)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith("metal-tc run unavailable: ")
    assert completed.stdout == "torch not imported\n"
    assert not output.parent.exists()


def test_a_tampered_file_exits_1_and_a_bad_input_exits_2(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert run(tmp_path, output) == EXIT_OK
    text = output.read_text(encoding="utf-8")
    output.write_text(text.replace('"warnings":0', '"warnings":1'), encoding="utf-8")

    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "rejected" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    config, ceilings = inputs(tmp_path)
    argv = ["run", "--config", str(config), "--ceilings", str(ceilings)]
    assert main([*argv, "--output", str(ceilings)]) == EXIT_INVALID
    assert main([*argv, "--output", str(config)]) == EXIT_INVALID


def test_a_report_file_must_be_canonical(tmp_path):
    report = synthetic_report(tmp_path)
    path = tmp_path / "pretty.json"
    save_report(path, report)
    load_report(path)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    with pytest.raises(MetalTcError, match="canonical"):
        load_report(path)


def test_the_parser_refuses_abbreviations():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--conf", "x"])


def test_the_verify_command_runs_where_torch_cannot_be_imported(tmp_path):
    output = tmp_path / "report.json"
    assert run(tmp_path, output) == EXIT_OK
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_tc_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, "verify", "--report", str(output)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith("verified_metal_tc_synthetic_timing=")
