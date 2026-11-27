"""Tests of the Day 82 command line: model, compile and verify, with exit codes.

The compile tests inject the SYNTHETIC backend of ``test_triton_tc_report``,
whose texts are hand-built and not compiler output. A fresh interpreter with
triton hidden shows that ``compile`` exits 3 and writes nothing. No test
imports triton.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import triton_tc_cli
from ai_journey.triton_tc_compile import CompiledVariant
from test_triton_tc_report import SyntheticTcBackend

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "day-82-triton-tc.json"
CLOCK = {"clock": lambda: 0.0, "load_average": lambda: (1.0, 2.0, 3.0)}
WITHOUT_TRITON = """
import sys
sys.modules["triton"] = None
from ai_journey.triton_tc_cli import main
sys.exit(main(sys.argv[1:]))
"""


def run(argv, capsys, **keywords):
    code = triton_tc_cli.main(argv, **{**CLOCK, **keywords})
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_model_prints_every_device_and_size_and_verifies_its_report(tmp_path, capsys):
    report = tmp_path / "model.json"
    code, lines, _ = run(
        ["model", "--config", str(CONFIG), "--output", str(report)], capsys
    )

    assert code == 0
    assert len(lines) == 1 + 2 * (1 + 5) + 1
    assert lines[0].startswith("triton_tc model label=declared_model_not_a_measurement")
    assert (
        "triton_tc model_row device=a100-sxm4-80gb n=512 fp32_us=13.766 "
        "tf32_us=1.721 bf16_us=1.029 bound=compute,compute,memory "
        "ratio=13.38:1.67:1 memory_bound=fp16,bf16"
    ) in lines
    assert any(
        "device=h100-sxm5-80gb n=2048 " in line and "ratio=14.77:2.00:1" in line
        for line in lines
    )
    assert lines[-1].endswith("gpu=false ptx_executed=false measured=false")
    assert json.loads(report.read_text("ascii"))["label"] == "declared_model"

    code, lines, _ = run(
        ["verify", "--report", str(report), "--config", str(CONFIG)], capsys
    )
    assert code == 0
    assert lines[0].startswith("verified_triton_tc_model=")
    assert lines[0].endswith("texts_recounted=false")


def test_compile_with_a_synthetic_backend_writes_a_report_and_a_dump(tmp_path, capsys):
    report, dump = tmp_path / "compile.json", tmp_path / "dump"
    argv = ["compile", "--config", str(CONFIG), "--output", str(report)]
    code, lines, _ = run(
        [*argv, "--dump", str(dump)], capsys, backend=SyntheticTcBackend()
    )

    assert code == 0
    assert len(lines) == 8 + 1 + 1
    assert lines[0].startswith(
        "triton_tc compile variant=fp32-ieee target=sm_80 observed=scalar_fma"
    )
    assert any(
        "variant=fp32-tf32 target=sm_90 observed=mma_sync" in line
        and line.endswith("note=mma_sync_on_sm_90")
        for line in lines
    )
    assert lines[8] == (
        "triton_tc compile_summary backend=injected triton=synthetic "
        "accepted=8/8 mma_sync_on_sm_90=fp32-tf32"
    )
    assert lines[9].startswith("triton_tc_synthetic=")
    assert (dump / "index.json").is_file()
    assert (dump / "bf16" / "sm_90" / "ptx.ptx").is_file()

    verify = ["verify", "--report", str(report), "--config", str(CONFIG)]
    code, lines, _ = run([*verify, "--dump", str(dump)], capsys)
    assert (code, lines[0].endswith("texts_recounted=true")) == (0, True)


def test_a_compile_that_breaks_the_rule_exits_1_and_keeps_its_report(tmp_path, capsys):
    class Scalar(SyntheticTcBackend):
        """SYNTHETIC: every variant gets the scalar text, so the rule fails."""

        def compile(self, variant, target):
            plain = super().compile(type(variant)("fp32-ieee", "fp32", "ieee"), target)
            return CompiledVariant(
                variant.name, target, plain.ttgir, plain.ptx, 65536, 4, 3
            )

    report = tmp_path / "rejected.json"
    argv = ["compile", "--config", str(CONFIG), "--output", str(report)]
    code, _, error = run(argv, capsys, backend=Scalar())

    assert code == 1
    assert "2 of 8 compiles pass the instruction-class rule" in error
    assert json.loads(report.read_text("ascii"))["compile"]["summary"]["accepted"] == 2


def test_compile_without_triton_exits_3_before_it_writes_anything(tmp_path):
    report, dump = tmp_path / "out" / "compile.json", tmp_path / "dump"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            WITHOUT_TRITON,
            "compile",
            "--config",
            str(CONFIG),
            "--output",
            str(report),
            "--dump",
            str(dump),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 3, completed.stderr
    assert "triton-tc compile unavailable: triton cannot be imported" in (
        completed.stderr
    )
    assert "(triton_not_importable)" in completed.stderr
    assert completed.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_the_module_runs_as_a_command_without_triton():
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_TRITON, "model", "--config", str(CONFIG)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1].startswith("triton_tc_model=")


def test_verify_rejects_a_changed_report_and_refuses_unreadable_inputs(
    tmp_path, capsys
):
    report = tmp_path / "model.json"
    run(["model", "--config", str(CONFIG), "--output", str(report)], capsys)
    value = json.loads(report.read_text("ascii"))
    value["summary"]["compile"] = {}
    report.write_text(json.dumps(value), encoding="ascii")
    verify = ["verify", "--report", str(report), "--config", str(CONFIG)]

    code, _, error = run(verify, capsys)
    assert code == 1
    assert "triton-tc verify rejected: report digest does not match" in error
    code, _, error = run(
        ["verify", "--report", str(tmp_path / "missing.json"), "--config", str(CONFIG)],
        capsys,
    )
    assert code == 2
    assert "triton-tc verify failed" in error


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["model", "--config", str(CONFIG), "--output", str(CONFIG)], "alias"),
        (
            [
                "compile",
                "--config",
                str(CONFIG),
                "--output",
                "{tmp}/r.json",
                "--dump",
                "{tmp}",
            ],
            "inside the dump folder",
        ),
        (["model", "--config", "{tmp}/missing.json"], "config must be"),
    ],
)
def test_refused_outputs_and_inputs_exit_2_before_any_work(
    argv, message, tmp_path, capsys
):
    words = [word.replace("{tmp}", str(tmp_path)) for word in argv]
    backend = SyntheticTcBackend()
    code, lines, error = run(words, capsys, backend=backend)

    assert code == 2
    assert message in error
    assert backend.calls == []
    assert lines == []
