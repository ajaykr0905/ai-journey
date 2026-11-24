"""Tests of the Day 81 command line: model, check and verify, and exit codes."""

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_journey import triton_flash_cli
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_flash_memory import MEMORY_TABLE
from test_triton_flash_correctness import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-81-triton-flash.json"
# A None entry in sys.modules makes every import of that name fail.
WITHOUT = """
import sys
for name in ("triton", "torch"):
    sys.modules[name] = None
from ai_journey.triton_flash_cli import main
sys.exit(main(sys.argv[1:]))
"""


def small_plan(tmp_path):
    plan = {
        "schema_version": 1,
        "name": "small-flash",
        "seed": 81,
        "cases": [
            {"batch_heads": 2, "sequence": 17, "head_dim": 16, "causal": True},
            {"batch_heads": 1, "sequence": 40, "head_dim": 32, "causal": False},
        ],
        "configs": ["m16_n16_w1_s1"],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def check(tmp_path, output, backend=None):
    return triton_flash_cli.main(
        ["check", "--config", str(small_plan(tmp_path)), "--output", str(output)],
        backend=backend or StandInBackend(),
        clock=lambda: 1795478400.0,
        load_average=lambda: (2.0, 1.5, 1.0),
    )


def test_model_prints_every_row_and_says_it_is_not_measured(capsys):
    assert triton_flash_cli.main(["model"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(MEMORY_TABLE) + 1
    assert lines[0].startswith(
        "memory n=512 d=64 naive_bytes=2621440 flash_bytes=528384 naive_mib=2.50"
    )
    assert "n=8192 d=128" in lines[-2] and "naive_mib=528.00" in lines[-2]
    assert lines[-1].startswith("model=declared measured=false gpu=false")
    assert "not a measured GPU footprint" in lines[-1]


def test_check_with_a_stand_in_writes_a_synthetic_report_that_verifies(
    tmp_path, capsys
):
    output = tmp_path / "out" / "report.json"

    assert check(tmp_path, output) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("synthetic_backend correctness bh=2 n=17 d=16")
    assert lines[-1].startswith("triton_flash_synthetic=")
    assert "within_bound=4/4" in lines[-1] and "load_before=2.00,1.50,1.00" in lines[-1]
    report = json.loads(output.read_text(encoding="ascii"))
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"

    code = triton_flash_cli.main(
        ["verify", "--report", str(output), "--config", str(small_plan(tmp_path))]
    )
    assert code == 0
    assert capsys.readouterr().out.startswith("verified_triton_flash_synthetic=")
    assert (
        triton_flash_cli.main(
            ["verify", "--report", str(output), "--config", str(PINNED)]
        )
        == 1
    )


def test_verify_rejects_a_tampered_report_with_exit_1(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert check(tmp_path, output) == 0
    report = json.loads(output.read_text(encoding="ascii"))
    report["summary"]["runs"] = 5
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    capsys.readouterr()

    assert triton_flash_cli.main(["verify", "--report", str(output)]) == 1
    assert "rejected: report digest" in capsys.readouterr().err


def test_bad_inputs_exit_2_and_write_nothing(tmp_path, capsys):
    plan = small_plan(tmp_path)

    assert check(tmp_path, plan) == 2
    assert json.loads(plan.read_text(encoding="utf-8"))["name"] == "small-flash"
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}', encoding="utf-8")
    code = triton_flash_cli.main(
        ["check", "--config", str(bad), "--output", str(tmp_path / "x.json")],
        backend=StandInBackend(),
    )
    assert code == 2
    assert not (tmp_path / "x.json").exists()
    assert triton_flash_cli.main(["verify", "--report", str(tmp_path / "no")]) == 2
    assert check(tmp_path, tmp_path / "f.json", StandInBackend("skip_rescale")) == 2
    assert not (tmp_path / "f.json").exists()
    assert "times its bound" in capsys.readouterr().err


def test_check_without_triton_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "missing" / "report.json"
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT, "check", "--config", str(PINNED)]
        + ["--output", str(output)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert result.returncode == 3, result.stderr
    assert "triton-flash check unavailable: triton cannot be imported" in result.stderr
    assert "(triton_not_importable)" in result.stderr
    assert result.stdout == ""
    assert not output.parent.exists()


def test_model_and_verify_need_no_triton(tmp_path):
    output = tmp_path / "report.json"
    assert check(tmp_path, output) == 0
    for argv in (["model"], ["verify", "--report", str(output)]):
        result = subprocess.run(
            [sys.executable, "-c", WITHOUT, *argv],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert result.returncode == 0, result.stderr
