"""Tests of the loc, check and verify commands and of their exit codes.

The commands run with the NumPy stand-in on any host. Two tests start a fresh
process: one hides triton, so ``check`` must exit 3 and write nothing, and one
runs the real interpreter, so it skips on a host without triton.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import triton_matmul_cli
from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.triton_matmul_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
)
from test_triton_matmul_correctness import StandInBackend
from test_triton_matmul_report import pinned_impostor

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-79-triton-matmul.json"
HAS_TRITON = importlib.util.find_spec("triton") is not None
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="no triton wheel here")
WITHOUT_TRITON = """
import sys
sys.modules["triton"] = None
from ai_journey.triton_matmul_cli import main
sys.exit(main(sys.argv[1:]))
"""
SMALL_CONFIG = {
    "schema_version": 1,
    "name": "triton-matmul-cli",
    "seed": 74,
    "correctness": {"shapes": [[17, 33, 9], [31, 47, 129]]},
    "configs": ["m128_n32_k32_g8_w4_s4", "m64_n32_k32_g8_w2_s5"],
}
TRITON_LOC = (
    "loc lab=day-79 language=triton text=matmul_kernel kernel=33 host=9 other=2 "
    "comment=10 blank=4 "
    "sha256=9eb35e273b0fd229e83e223861ef7ce13c4f15ea46d2694baf13920851a31d56"
)


def write_config(path, mapping=None):
    path.write_text(json.dumps(mapping or SMALL_CONFIG), encoding="utf-8")
    return path


def check_argv(config, output):
    return ["check", "--config", str(config), "--output", str(output)]


def stand_in_main(argv, backend=None):
    return main(
        argv,
        backend=backend or StandInBackend(),
        clock=lambda: 1795300000.0,
        load_average=lambda: (1.0, 2.0, 3.0),
    )


def fresh_process(code, *arguments, environment=None):
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **(environment or {})},
    )


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == PROG == "ai-journey-triton-matmul"
    parsed = parser.parse_args(["check", "--config", "c.json", "--output", "r.json"])
    assert (parsed.command, str(parsed.config), str(parsed.output)) == (
        "check",
        "c.json",
        "r.json",
    )
    assert parser.parse_args(["verify", "--report", "r.json"]).config is None
    for argv in ([], ["check", "--config", "c.json"], ["check", "--conf", "c.json"]):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)


def test_loc_prints_one_line_for_each_text(capsys):
    assert main(["loc"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 19
    assert lines[0] == TRITON_LOC
    assert lines[2].startswith(
        "loc lab=day-76 language=cuda text=tiled_matmul_t16_kernel kernel=24 host=9 "
    )
    assert lines[-1].startswith(
        "loc lab=day-77 language=metal text=matmul_s7_interior kernel=420 host=none "
    )


def test_check_then_verify_print_one_digest_line(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "out" / "report.json"

    assert stand_in_main(check_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 3
    assert re.fullmatch(
        r"synthetic_backend correctness m=17 n=33 k=9 bound=5\.36e-07 configs=2 "
        r"normal_max_ratio=0\.\d{3} exact_bits=2/2",
        lines[0],
    )
    assert lines[1].startswith("synthetic_backend correctness m=31 n=47 k=129 ")
    assert re.fullmatch(
        r"triton_matmul_synthetic=[0-9a-f]{64} plan=triton-matmul-cli cases=2 "
        r"configs=2 within_bound=4/4 exact_bits=4/4 max_bound_ratio=0\.\d{3} "
        r"triton=none torch=none machine=x86_64 load_before=1\.00,2\.00,3\.00 "
        r"load_after=1\.00,2\.00,3\.00 kernel_lines=33 host_lines=9 gpu=false "
        r"autotuned=false timing=false",
        lines[2],
    )
    assert output.read_bytes().endswith(b"\n")
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[2]}\n"


def test_check_without_triton_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "report.json"

    completed = fresh_process(WITHOUT_TRITON, *check_argv(PINNED, output))

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("triton-matmul check unavailable: ")
    assert "(triton_not_importable)" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_check_with_the_interpreter_turned_off_exits_3(tmp_path):
    output = tmp_path / "new" / "report.json"
    code = (
        "import sys\nfrom ai_journey.triton_matmul_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )

    completed = fresh_process(
        code, *check_argv(PINNED, output), environment={"TRITON_INTERPRET": "0"}
    )

    reason = "interpreter_disabled" if HAS_TRITON else "triton_not_importable"
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_check_never_writes_over_its_config_or_after_a_broken_rule(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()

    assert stand_in_main(check_argv(config, config)) == EXIT_INVALID
    assert "must not alias the config" in capsys.readouterr().err
    assert config.read_bytes() == before
    output = tmp_path / "broken" / "report.json"
    broken = StandInBackend("skip_k_tail")
    assert stand_in_main(check_argv(config, output), broken) == EXIT_INVALID
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exceeds the bound" in captured.err
    assert not (tmp_path / "broken").exists()
    bad = write_config(tmp_path / "bad.json", {**SMALL_CONFIG, "seed": True})
    assert stand_in_main(check_argv(bad, output)) == EXIT_INVALID
    assert "seed must be an integer" in capsys.readouterr().err


def test_check_refuses_an_injected_load_average_for_the_interpreter(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(triton_matmul_cli, "InterpreterBackend", pinned_impostor)
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "out" / "report.json"

    argv = check_argv(config, output)
    assert main(argv, load_average=lambda: (1.0, 2.0, 3.0)) == EXIT_INVALID
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "host clock" in captured.err
    assert not (tmp_path / "out").exists()
    # With the host clock and load average, the same backend gets its label.
    assert main(argv) == EXIT_OK
    summary = capsys.readouterr().out.splitlines()[-1]
    assert summary.startswith("triton_matmul_interpreter=")


def test_verify_rejects_tampered_reformatted_and_unbound_reports(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    assert stand_in_main(check_argv(config, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    changed = {**report, "label": "triton_cpu_interpreter_check"}
    tampered.write_bytes(canonical_json_bytes(changed) + b"\n")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    other = write_config(tmp_path / "other.json", {**SMALL_CONFIG, "seed": 79})
    for argv in (
        ["verify", "--report", str(tampered)],
        ["verify", "--report", str(pretty)],
        ["verify", "--report", str(output), "--config", str(other)],
    ):
        assert main(argv) == EXIT_REJECTED
        assert capsys.readouterr().err.startswith("triton-matmul verify rejected: ")
    for argv in (
        ["verify", "--report", str(tmp_path / "missing.json")],
        ["verify", "--report", str(output), "--config", str(tmp_path / "none.json")],
    ):
        assert main(argv) == EXIT_INVALID
        assert capsys.readouterr().err.startswith("triton-matmul verify failed: ")


def test_the_module_runs_as_a_command_with_usage_errors_as_code_2():
    command = [sys.executable, "-m", "ai_journey.triton_matmul_cli"]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    listed = subprocess.run(
        [*command, "loc"], capture_output=True, text=True, env=environment, check=False
    )
    assert listed.returncode == EXIT_OK
    assert listed.stdout.splitlines()[0] == TRITON_LOC
    usage = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    assert usage.returncode == EXIT_INVALID
    assert usage.stderr.startswith(f"usage: {PROG}")


@requires_triton
def test_check_runs_the_interpreter_in_a_fresh_process(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    code = (
        "import sys\nfrom ai_journey.triton_matmul_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )

    completed = fresh_process(code, *check_argv(config, output))

    assert completed.returncode == EXIT_OK, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0].startswith("triton_interpreter_check correctness m=17 n=33 k=9 ")
    assert lines[-1].startswith("triton_matmul_interpreter=")
    assert " within_bound=4/4 exact_bits=4/4 " in lines[-1]
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
