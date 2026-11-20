import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.gpu_scan_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
    short_name,
)
from ai_journey.gpu_scan_io import load_report
from ai_journey.gpu_scan_plan import PLAN
from ai_journey.gpu_scan_source import KERNELS, S2_SIMD
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_scan_config import make_config_mapping
from test_gpu_scan_correctness import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-78-gpu-scan.json"
LOAD = (1.5, 2.25, 3.0)
NUMBER = r"\d+(?:\.\d+)?(?:e[-+]\d+)?"
RATE = r"\d+\.\d{2}"
RATIO = r"(?:\d+\.\d{3}:\w+|none)"
FRACTION = r"(?:\d+\.\d{3}|none)"
SIDES = [short_name(name) for name in KERNELS] + ["torch"]
CASE = (
    r"correctness length=\d+ pattern=(?:standard_normal|small_integers) "
    r"levels=\d+(?:,\d+)* "
    + " ".join(
        rf"{side}_error={NUMBER} {side}_bits=(?:replay|exact|other|unknown) "
        rf"{side}_deterministic=(?:true|false)"
        for side in SIDES
    )
)
POINT = r"benchmark length=\d+ dispatches=\d+ " + " ".join(
    rf"{side}_gbps={RATE} {side}_iqr={RATE}-{RATE}" for side in (*SIDES, "copy")
)
SPEEDUP = (
    rf"speedup length=\d+ step=S\d kernel=\w+ gbps=(?:{RATE}|none) "
    rf"to_base={RATIO} to_s0={RATIO} to_torch={RATIO} copy_fraction={FRACTION} "
    rf"modeled_copy_fraction={FRACTION} held=(?:true|false|none)"
)
BEST = (
    rf"best length=\d+ kernel=\w+ to_torch={RATIO} torch_gbps={RATE} "
    rf"torch_copy_fraction=\d+\.\d{{3}} copy_gbps={RATE}"
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.gpu_scan_cli import main
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


def run_argv(config, output):
    return ["run", "--config", str(config), "--output", str(output)]


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

    assert parser.prog == "ai-journey-gpu-scan"
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
    assert [short_name(name) for name in (*KERNELS, "torch_cumsum", "stream_copy")] == [
        "s0_hillis_steele",
        "s1_blelloch",
        "s2_simd",
        "torch",
        "copy",
    ]


def test_a_stand_in_run_prints_the_plan_and_every_number(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    label = "synthetic_timing"
    steps = len(PLAN)
    assert len(lines) == steps + 1 + 3 + 2 + 2 * (steps + 1) + 1 + 1
    assert lines[0] == (
        f"{label} plan step=S0 kernel=s0_hillis_steele base=torch "
        "predictions=65536:slower,1048576:slower,16777216:slower,5000003:slower"
    )
    assert lines[2] == (
        f"{label} plan step=S2 kernel=s2_simd base=s0_hillis_steele "
        "predictions=65536:no_clear_difference,1048576:faster,16777216:faster,"
        "5000003:faster"
    )
    assert lines[steps] == (
        f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    for line in lines[steps + 1 : steps + 4]:
        assert re.fullmatch(f"{label} {CASE}", line), line
    assert "s0_hillis_steele_bits=replay" in lines[steps + 1]
    assert "s2_simd_bits=unknown" in lines[steps + 1]
    assert "s1_blelloch_bits=exact" in lines[steps + 2]
    for line in lines[steps + 4 : steps + 6]:
        assert re.fullmatch(f"{label} {POINT}", line), line
    speedups = lines[steps + 6 : -2]
    assert [re.fullmatch(f"{label} {BEST}", line) is not None for line in speedups] == (
        [False] * steps + [True]
    ) * 2
    for line in speedups:
        assert re.fullmatch(f"{label} ({SPEEDUP}|{BEST})", line), line
    assert re.fullmatch(
        f"{label} warnings count=0 distinct=0 torch_imported_before_capture=(true|false)",
        lines[-2],
    )
    assert lines[-1] == (
        f"gpu_scan_synthetic_timing={report['report_sha256']} "
        "correctness_cases=3 benchmark_points=2 refused=none hypotheses_held=0/0 "
        f"plan={report['plan']['sha256'][:12]} "
        "scope=synthetic_backend_or_clock load_before=1.50,2.25,3.00 "
        "load_after=1.50,2.25,3.00 measured=false cuda=false nvidia=false "
        "warp_shuffles=false hardware_counters=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_a_refused_kernel_is_printed_and_named_in_the_summary(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    backend = StandInBackend(widths={S2_SIMD: 16})

    assert stand_in_main(run_argv(config, output), backend=backend) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert lines[len(PLAN) + 1] == (
        "synthetic_timing refused kernel=scan_s2_simd reason=simd_width_is_not_32 "
        "allowed_threads=1024 static_threadgroup_bytes=128 simd_width=16"
    )
    assert "s2_simd_error" not in "\n".join(lines)
    assert any("kernel=s2_simd gbps=none to_base=none" in line for line in lines)
    assert " refused=scan_s2_simd " in lines[-1]


def test_a_host_without_torch_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "report.json"

    completed = without_torch(*run_argv(PINNED, output))

    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("gpu-scan run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_rejects_tampered_or_reformatted_reports_and_reads_strictly(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    changed = {**report, "label": "host-specific Apple GPU (Metal) measurement"}
    tampered.write_bytes(canonical_json_bytes(changed) + b"\n")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 79})

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    assert (
        main(["verify", "--report", str(output), "--config", str(other)])
        == EXIT_REJECTED
    )
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    assert main(["verify", "--report", str(tmp_path)]) == EXIT_INVALID


def test_verify_never_imports_torch(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, output)) == EXIT_OK

    completed = without_torch(
        "verify", "--report", str(output), "--config", str(config)
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout.startswith("verified_gpu_scan_synthetic_timing=")


def test_run_refuses_an_aliasing_output_a_device_error_and_a_spent_budget(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()
    assert stand_in_main(run_argv(config, config)) == EXIT_INVALID
    assert config.read_bytes() == before

    class FailingBackend(StandInBackend):
        def scan(self, name, values):
            raise RuntimeError("MPS backend out of memory")

    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, output), backend=FailingBackend()) == 2
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    short = write_config(tmp_path / "short.json", mapping)
    clock = scripted_clock(step=10**9)
    assert stand_in_main(run_argv(short, output), clock=clock) == EXIT_INVALID
    bad = write_config(tmp_path / "bad.json", {**mapping, "plan_sha256": "0" * 64})
    assert stand_in_main(run_argv(bad, output)) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "out of memory" in errors and "exceeded max_runtime_seconds=1" in errors
    assert "does not name the declared plan" in errors
    assert not output.exists()
