import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_reduce_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
    short_name,
)
from ai_journey.gpu_reduce_io import load_report, save_report
from ai_journey.gpu_reduce_plan import PLAN
from ai_journey.gpu_reduce_source import STEP_KERNELS, GpuReduceError
from ai_journey.gpu_reduce_verify import GpuReduceVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_reduce_config import make_config_mapping
from test_gpu_reduce_correctness import StandInBackend
from test_gpu_reduce_speedup import write_ceiling

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-78-gpu-reduce.json"
LOAD = (1.5, 2.25, 3.0)
NUMBER = r"\d+(?:\.\d+)?(?:e[-+]\d+)?"
RATE = r"\d+\.\d"
RATIO = r"(?:\d+\.\d{3}:\w+|none)"
SHORTS = [short_name(name) for name in STEP_KERNELS]
CASE = r"correctness pattern=\w+ n=\d+ exact=(?:true|false) " + " ".join(
    rf"{short}={NUMBER}:(?:same|changed)" for short in SHORTS
)
POINT = r"benchmark n=\d+ " + " ".join(
    rf"{short}_gbs={RATE} {short}_iqr={RATE}-{RATE}" for short in (*SHORTS, "torch")
)
SPEEDUP = (
    rf"speedup n=\d+ step=R\d kernel=\w+ gbs={RATE} ceiling_fraction=\d+\.\d{{3}} "
    rf"torch_fraction=\d+\.\d{{3}} to_base={RATIO} to_r0={RATIO} "
    r"held=(?:true|false)"
)
BEST = (
    rf"best n=\d+ kernel=\w+ gbs={RATE} ceiling_fraction=\d+\.\d{{3}} "
    rf"to_r0={RATIO} torch_gbs={RATE} torch_ceiling_fraction=\d+\.\d{{3}}"
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.gpu_reduce_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(path, mapping=None):
    path.write_text(json.dumps(mapping or make_config_mapping()), encoding="utf-8")
    return path


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def stand_in_main(argv, backend=None, clock=None):
    return main(
        argv,
        backend=backend or StandInBackend(),
        clock=clock or scripted_clock(),
        load_average=lambda: LOAD,
    )


def run_argv(config, ceilings, output):
    return [
        "run",
        "--config",
        str(config),
        "--ceilings",
        str(ceilings),
        "--output",
        str(output),
    ]


def without_torch(*arguments):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT_TORCH, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


@pytest.fixture
def inputs(tmp_path):
    return write_config(tmp_path / "config.json"), write_ceiling(tmp_path)


def test_the_parser_requires_explicit_commands_and_paths():
    parser = build_parser()

    assert parser.prog == PROG == "ai-journey-gpu-reduce"
    parsed = parser.parse_args(run_argv("c.json", "g.json", "r.json"))
    assert (parsed.command, str(parsed.ceilings), str(parsed.output)) == (
        "run",
        "g.json",
        "r.json",
    )
    verify = parser.parse_args(["verify", "--report", "r.json"])
    assert verify.config is None and verify.ceilings is None
    for argv in (
        [],
        ["run", "--config", "c.json", "--output", "r.json"],
        ["run", "--conf", "c.json", "--ceilings", "g.json", "--output", "r.json"],
    ):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)
    assert [short_name(name) for name in ("reduce_r4_simd_sum", "torch_sum")] == [
        "r4_simd_sum",
        "torch",
    ]


def test_a_stand_in_run_prints_the_plan_and_every_number(inputs, tmp_path, capsys):
    config, ceilings = inputs
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    label = "synthetic_timing"
    steps = len(PLAN)
    assert len(lines) == steps + 2 + 6 + 2 + 2 * (steps + 1) + 1 + 1 + 1
    assert lines[0] == (
        f"{label} plan step=R0 kernel=r0_interleaved base=torch small=faster|"
        "no_clear_difference medium=slower large=slower"
    )
    assert lines[3] == (
        f"{label} plan step=R3 kernel=r3_simd_shuffle base=r2_first_add "
        "small=no_clear_difference medium=no_clear_difference|faster "
        "large=no_clear_difference"
    )
    assert lines[steps] == (
        f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    assert lines[steps + 1] == (
        f"{label} ceiling gb_per_second=82.28 kernel=stream_read_reduce "
        "buffer_bytes=4194304 measured=true "
        f"same_environment={'true' if report['ceiling']['same_environment'] else 'false'} "
        "report=6b784c28e36b load_before=9.48,9.40,8.75 load_after=9.48,9.40,8.75"
    )
    for line in lines[steps + 2 : steps + 8]:
        assert re.fullmatch(f"{label} {CASE}", line), line
    for line in lines[steps + 8 : steps + 10]:
        assert re.fullmatch(f"{label} {POINT}", line), line
    speedups = lines[steps + 10 : -3]
    assert [re.fullmatch(f"{label} {BEST}", line) is not None for line in speedups] == (
        [False] * steps + [True]
    ) * 2
    for line in speedups:
        assert re.fullmatch(f"{label} ({SPEEDUP}|{BEST})", line), line
    assert lines[-3] == (
        f"{label} same_order kernel=r3_simd_shuffle as=r2_first_add "
        "cases_with_other_bits=none points_with_other_bits=none"
    )
    assert re.fullmatch(
        f"{label} warnings count=0 distinct=0 torch_imported_before_capture=(true|false)",
        lines[-2],
    )
    assert lines[-1] == (
        f"gpu_reduce_synthetic_timing={report['report_sha256']} "
        "correctness_cases=6 benchmark_points=2 hypotheses_held=12/12 "
        f"plan={report['plan']['sha256'][:12]} ceiling=6b784c28e36b "
        "fraction_measured=false scope=synthetic_backend_or_clock "
        "load_before=1.50,2.25,3.00 load_after=1.50,2.25,3.00 measured=false "
        "cuda=false nvidia=false warp_level=false hardware_counters=false"
    )
    argv = ["verify", "--report", str(output), "--config", str(config)]
    assert main([*argv, "--ceilings", str(ceilings)]) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_a_host_without_torch_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "report.json"
    ceilings = write_ceiling(tmp_path)

    completed = without_torch(*run_argv(PINNED, ceilings, output))

    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("gpu-reduce run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_rejects_tampered_reformatted_or_foreign_reports(
    inputs, tmp_path, capsys
):
    config, ceilings = inputs
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    changed = {**report, "label": "host-specific Apple GPU (Metal) measurement"}
    tampered.write_bytes(canonical_json_bytes(changed) + b"\n")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 79})
    foreign = json.loads(CEILING_JSON)
    foreign["summary"] = {**foreign["summary"], "bank_points": 5}
    bad_ceilings = tmp_path / "bad" / "gpu-ceilings.json"
    bad_ceilings.parent.mkdir()
    bad_ceilings.write_bytes(canonical_json_bytes(foreign) + b"\n")

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    argv = ["verify", "--report", str(output)]
    assert main([*argv, "--config", str(other)]) == EXIT_REJECTED
    assert main([*argv, "--ceilings", str(bad_ceilings)]) == EXIT_INVALID
    assert "ceiling report was refused" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == 2
    with pytest.raises(GpuReduceVerificationError, match="digest"):
        load_report(tampered)
    with pytest.raises(GpuReduceError, match="non-symlink"):
        load_report(tmp_path)


def test_run_refuses_aliasing_outputs_a_device_error_and_a_spent_budget(
    inputs, tmp_path, capsys
):
    config, ceilings = inputs
    before = (config.read_bytes(), ceilings.read_bytes())
    assert stand_in_main(run_argv(config, ceilings, config)) == EXIT_INVALID
    assert stand_in_main(run_argv(config, ceilings, ceilings)) == EXIT_INVALID
    assert (config.read_bytes(), ceilings.read_bytes()) == before

    class FailingBackend(StandInBackend):
        def run(self, name, values):
            raise RuntimeError("MPS backend out of memory")

    output = tmp_path / "out" / "report.json"
    argv = run_argv(config, ceilings, output)
    assert stand_in_main(argv, backend=FailingBackend()) == EXIT_INVALID
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    short = write_config(tmp_path / "short.json", mapping)
    clock = scripted_clock(step=10**9)
    assert stand_in_main(run_argv(short, ceilings, output), clock=clock) == 2
    errors = capsys.readouterr().err
    assert "alias the ceilings report" in errors
    assert "out of memory" in errors and "exceeded max_runtime_seconds=1" in errors
    assert not output.exists()


def test_save_report_verifies_before_it_writes(inputs, tmp_path):
    config, ceilings = inputs
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    report = json.loads(output.read_text(encoding="utf-8"))
    target = tmp_path / "copy.json"

    with pytest.raises(GpuReduceVerificationError):
        save_report(target, {**report, "measured": True})
    assert not target.exists()
    with pytest.raises(GpuReduceError, match="alias a protected input"):
        save_report(config, report, protected=(config,))


def test_the_module_runs_with_python_m_and_without_torch():
    completed = subprocess.run(
        [sys.executable, "-m", "ai_journey.gpu_reduce_cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    blocked = without_torch("--help")

    for result in (completed, blocked):
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(f"usage: {PROG} ")
        assert "{run,verify}" in result.stdout
