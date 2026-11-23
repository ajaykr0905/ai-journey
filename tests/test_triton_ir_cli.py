"""Tests of the Day 80 Triton IR command line: output lines, files and exit codes.

``check`` runs with a stand-in backend here, or in a fresh process with no
triton, where it must exit 3 and write nothing. One test runs the real
triton and skips cleanly when triton cannot be imported.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.triton_ir_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
)
from test_triton_ir_runtime import FixtureBackend

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "day80_triton_ir"
WITHOUT_TRITON = """
import sys
sys.modules["triton"] = None
from ai_journey.triton_ir_cli import main
sys.exit(main(sys.argv[1:]))
"""
CLOCK = {"clock": lambda: 0.0, "load_average": lambda: (1.0, 2.0, 3.0)}


def fresh_process(code, *arguments, environment=None):
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(environment or {})},
    )


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()
    for argv in (
        [],
        ["summarize"],
        ["stage-map"],
        ["check", "--fixtures", "x"],
        ["verify", "--report", "x"],
        ["summ", "--fixtures", "x"],
        ["summarize", "--fix", "x"],
    ):
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(argv)
        assert raised.value.code == EXIT_INVALID


def test_summarize_prints_the_counts_and_writes_a_report_that_verifies(
    tmp_path, capsys
):
    output = tmp_path / "out" / "report.json"
    argv = ["summarize", "--fixtures", str(FIXTURES), "--output", str(output)]
    assert main(argv, **CLOCK) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 26
    assert lines[1] == (
        "triton_ir ptx target=sm_80 version=8.8 address_size=64 entry=matmul_kernel "
        "params=14 reqntid=64 registers=pred:251,b32:2296,b64:480 "
        "shared_extern=global_smem instructions=3097 predicated=34"
    )
    assert lines[6] == (
        "triton_ir pipeline config=m32_n64_k32_g8_w2_s5 num_stages=5 buffers=4 "
        "shared_bytes=49152 prologue_groups=8 wait_in_loop=6 fma_per_thread=1024 "
        "mma=0"
    )
    assert lines[-1].startswith("triton_ir_recorded=")
    assert lines[-1].endswith(
        " ptx_instructions=3097 fma=1024 cp_async=252 mma=0 compiles=12 "
        "load_before=1.00,2.00,3.00 load_after=1.00,2.00,3.00 gpu=false "
        "ptx_executed=false"
    )
    assert main(["verify", "--report", str(output), "--fixtures", str(FIXTURES)]) == 0
    verified = capsys.readouterr().out
    assert verified == f"verified_{lines[-1]}\n"


def test_stage_map_prints_one_line_for_each_statement(capsys):
    assert main(["stage-map", "--fixtures", str(FIXTURES)]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 10
    assert lines[5] == (
        "stage_map line=42 key=dot ttir=tt.dot:1 ttgir=tt.dot:1 "
        "ptx=fma.rn.f32:1024 k_loop=fma.rn.f32:1024"
    )
    assert lines[-1].startswith("stage_map line=47 key=store ttir=tt.store:1 ")


def test_check_without_triton_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "report.json"
    completed = fresh_process(
        WITHOUT_TRITON, "check", "--fixtures", str(FIXTURES), "--output", str(output)
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("triton-ir check unavailable: ")
    assert "(triton_not_importable)" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_check_with_a_stand_in_backend_writes_a_synthetic_report_and_exits_1(
    tmp_path, capsys
):
    output = tmp_path / "check.json"
    argv = ["check", "--fixtures", str(FIXTURES), "--output", str(output)]
    assert main(argv, backend=FixtureBackend(), **CLOCK) == EXIT_REJECTED
    captured = capsys.readouterr()
    assert captured.err == (
        "triton-ir check rejected: 3 of 48 texts and 12 of 12 metadata records "
        "match the dump\n"
    )
    lines = captured.out.splitlines()
    assert lines[-2] == (
        "triton_ir regeneration backend=injected triton=stand-in texts=3/48 "
        "metadata=12/12 cubins_sm_80=0/6 cubins_sm_90=0/6"
    )
    assert lines[-1].startswith("triton_ir_synthetic=")
    report = json.loads(output.read_bytes())
    assert report["label"] == "synthetic_regeneration"
    assert main(["verify", "--report", str(output), "--fixtures", str(FIXTURES)]) == 0
    assert capsys.readouterr().out.startswith("verified_triton_ir_synthetic=")


def test_bad_inputs_and_outputs_exit_2_and_write_nothing(tmp_path, capsys):
    folder = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, folder)
    manifest = folder / "index.json"
    argv = ["summarize", "--fixtures", str(folder), "--output", str(manifest)]
    before = manifest.read_bytes()
    assert main(argv, **CLOCK) == EXIT_INVALID
    assert "alias the config" in capsys.readouterr().err
    assert manifest.read_bytes() == before
    (folder / "sm_80" / "m32_n64_k32_g8_w2_s5" / "ptx.ptx").write_text("x\n")
    output = tmp_path / "report.json"
    argv = ["summarize", "--fixtures", str(folder), "--output", str(output)]
    assert main(argv, **CLOCK) == EXIT_INVALID
    assert "does not match its manifest digest" in capsys.readouterr().err
    assert not output.exists()
    argv = ["check", "--fixtures", str(folder), "--output", str(tmp_path / "d")]
    (tmp_path / "d").mkdir()
    assert main(argv, backend=FixtureBackend(), **CLOCK) == EXIT_INVALID
    assert "regular non-symlink" in capsys.readouterr().err


def test_verify_rejects_a_tampered_or_reformatted_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    argv = ["summarize", "--fixtures", str(FIXTURES), "--output", str(output)]
    assert main(argv, **CLOCK) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_bytes())
    verify = ["verify", "--report", str(output), "--fixtures", str(FIXTURES)]
    output.write_text(json.dumps(report, indent=1) + "\n")
    assert main(verify) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    report["summary"]["mma"] = 1
    output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    assert main(verify) == EXIT_REJECTED
    assert capsys.readouterr().err.startswith("triton-ir verify rejected: ")
    assert (
        main(["verify", "--report", str(tmp_path / "no.json"), "--fixtures", "."]) == 2
    )
    assert capsys.readouterr().err.startswith("triton-ir verify failed: ")


def test_the_module_runs_as_a_command_with_usage_errors_as_code_2():
    command = [sys.executable, "-m", "ai_journey.triton_ir_cli"]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    mapped = subprocess.run(
        [*command, "stage-map", "--fixtures", str(FIXTURES)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert mapped.returncode == EXIT_OK, mapped.stderr
    assert mapped.stdout.splitlines()[0].startswith("stage_map line=36 key=k_loop ")
    usage = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    assert usage.returncode == EXIT_INVALID
    assert usage.stderr.startswith(f"usage: {PROG}")


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="triton is not importable on this host; it has no macOS wheel",
)
def test_check_regenerates_every_text_with_the_real_triton(tmp_path, capsys):
    output = tmp_path / "check.json"
    code = (
        "import sys\nfrom ai_journey.triton_ir_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    completed = fresh_process(
        code,
        "check",
        "--fixtures",
        str(FIXTURES),
        "--output",
        str(output),
        environment={"TRITON_INTERPRET": "0", "TRITON_DEFAULT_FP_FUSION": "1"},
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert " texts=48/48 metadata=12/12 " in completed.stdout
    assert main(["verify", "--report", str(output), "--fixtures", str(FIXTURES)]) == 0
    assert capsys.readouterr().out.startswith("verified_triton_ir_regenerated=")
