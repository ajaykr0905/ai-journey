import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The report records whether torch was imported before its warning capture.
# Import it here, so the answer is the same on macOS and on the Linux CI job.
import torch  # noqa: F401

from ai_journey import metal_flash_runtime
from ai_journey.kernel_metal_runtime import probe_metal
from ai_journey.metal_flash_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.metal_flash_io import load_report, save_report
from ai_journey.metal_flash_source import MetalFlashError
from test_metal_flash_emulator import FlashEmulator
from test_metal_flash_report import SMALL_CONFIG, synthetic_report
from test_metal_fused_ceiling import write_recorded
from test_metal_liger_emulator import ticking_clock

ROOT = Path(__file__).resolve().parents[1]


def inputs(directory):
    config = directory / "config.json"
    config.write_text(json.dumps(SMALL_CONFIG), encoding="utf-8")
    return config, write_recorded(directory)


def run(directory, output, **keywords):
    config, ceilings = inputs(directory)
    argv = ["run", "--config", str(config), "--ceilings", str(ceilings)]
    return main(
        [*argv, "--output", str(output)],
        backend=keywords.get("backend", FlashEmulator()),
        clock=ticking_clock(),
        load_average=lambda: (1.0, 2.0, 3.0),
    )


def test_a_run_with_a_stand_in_publishes_a_synthetic_report(tmp_path, capsys):
    output = tmp_path / "report.json"

    assert run(tmp_path, output) == EXIT_OK
    printed = capsys.readouterr().out
    assert "synthetic_timing plan step=f1_over_n0 first=f1 base=n0" in printed
    assert "synthetic_timing benchmark point=0 sequences=2 length=64" in printed
    assert "n0_peak_driver_mib=" in printed
    assert "metal_flash_synthetic_timing=" in printed
    assert "measured=false cuda=false nvidia=false ptx=false" in printed
    config, ceilings = inputs(tmp_path)
    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main([*argv, "--ceilings", str(ceilings)]) == EXIT_OK
    assert "verified_metal_flash_synthetic_timing=" in capsys.readouterr().out


def test_a_host_without_mps_exits_3_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def linux(*, system, import_module):
        return probe_metal(system=lambda: "Linux", import_module=import_module)

    monkeypatch.setattr(metal_flash_runtime, "probe_metal", linux)
    output = tmp_path / "report.json"

    assert run(tmp_path, output, backend=None) == EXIT_UNAVAILABLE
    assert not output.exists()
    assert "metal-flash run unavailable: Metal kernels run only on macOS" in (
        capsys.readouterr().err
    )


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

    with pytest.raises(MetalFlashError, match="canonical"):
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
        "from ai_journey.metal_flash_cli import main\n"
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
    assert completed.stdout.startswith("verified_metal_flash_synthetic_timing=")
