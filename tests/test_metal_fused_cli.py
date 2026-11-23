import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.kernel_metal_geometry import PipelineLimits
from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.metal_fused_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    PROG,
    build_parser,
    main,
    short_name,
)
from ai_journey.metal_fused_io import load_report
from ai_journey.metal_fused_plan import PLAN
from ai_journey.metal_fused_source import F3, STEP_VARIANTS
from test_metal_fused_benchmark import StepClock
from test_metal_fused_ceiling import write_recorded
from test_metal_fused_config import make_config_mapping
from test_metal_fused_correctness import StandInBackend, recorded_limits

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-79-metal-fused.json"
LOAD = (1.5, 2.25, 3.0)


def write_config(path, mapping=None):
    path.write_text(json.dumps(mapping or make_config_mapping()), encoding="utf-8")
    return path


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


def stand_in_main(argv, backend=None):
    return main(
        argv,
        backend=backend or StandInBackend(),
        clock=StepClock(),
        load_average=lambda: LOAD,
    )


def inputs(directory):
    return write_config(directory / "config.json"), write_recorded(directory)


def without_torch(*argv):
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from ai_journey.metal_fused_cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script, *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_the_parser_takes_two_commands_with_no_abbreviations():
    parser = build_parser()

    assert parser.prog == PROG == "ai-journey-metal-fused"
    for argv in (["run", "--config", "c"], ["verify"], ["verify", "--rep", "r"]):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)
    assert [short_name(name) for name in (*STEP_VARIANTS, "torch_softmax")] == [
        "u0",
        "f1",
        "f2",
        "f3",
        "torch",
    ]


def test_a_stand_in_run_prints_the_plan_and_every_number(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    label = "synthetic_timing"
    points = len(report["benchmark"]["points"])
    cases = len(report["correctness"]["cases"])
    assert len(lines) == len(PLAN) + 3 + cases + points + points * 5 + 1 + 1
    assert lines[0] == (
        f"{label} plan step=U0 kernel=u0 base=torch predicted=slower same_bits_as=none"
    )
    assert lines[1] == (
        f"{label} plan step=F1 kernel=f1 base=u0 predicted=faster same_bits_as=u0"
    )
    assert (
        lines[4] == f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    assert lines[5].startswith(
        f"{label} ceiling gb_per_second=82.28 kernel=stream_read_reduce "
        "buffer_bytes=4194304 report=6b784c28e36b measured=true"
    )
    assert re.fullmatch(
        rf"{label} exp_probe max_ulps=\d\.\d{{3}} normal_results=\d+ "
        r"tiny_results=\d+ tiny_nonzero=\d+ holds=true",
        lines[6],
    )
    assert lines[7].startswith(
        f"{label} correctness case=0 rows=3 shortest=1 longest=33 group=32 "
        "inputs=uniform_80 naive_finite=true u0_ratio="
    )
    assert lines[7].endswith("f1_bits_as_u0=true")
    assert " naive_finite=false " in lines[8]
    assert lines[9].startswith(
        f"{label} benchmark rows=8 cols=64 group=32 calls_per_sample=1 u0_gbps="
    )
    assert lines[11].startswith(f"{label} speedup rows=8 cols=64 step=U0 kernel=u0 ")
    assert lines[15].startswith(f"{label} best rows=8 cols=64 kernel=")
    assert lines[-2].startswith(f"{label} warnings count=0 distinct=0 ")
    assert lines[-1].startswith(
        f"metal_fused_synthetic_timing={report['report_sha256']} "
    )
    assert lines[-1].endswith(
        "measured=false cuda=false nvidia=false triton_gpu=false hardware_counters=false"
    )


def test_verify_prints_the_same_summary_with_or_without_its_inputs(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    summary = capsys.readouterr().out.splitlines()[-1]

    for extra in ([], ["--config", str(config)], ["--ceilings", str(ceilings)]):
        assert main(["verify", "--report", str(output), *extra]) == EXIT_OK
        assert capsys.readouterr().out == f"verified_{summary}\n"


def test_a_refused_step_is_printed_and_named_in_the_summary(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    backend = StandInBackend(
        limits=recorded_limits(**{F3: PipelineLimits(1024, 16, 256)})
    )

    assert stand_in_main(run_argv(config, ceilings, output), backend) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert lines[7] == (
        "synthetic_timing refused step=f3 reason=pipeline_simd_width_is_not_32 "
        "required_threads=32"
    )
    assert any("kernel=f3 gbps=none to_base=none" in line for line in lines)
    assert " refused=f3 " in lines[-1]


def test_verify_rejects_tampered_reformatted_or_foreign_reports(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    changed = {**report, "label": "host-specific Apple GPU (Metal) measurement"}
    tampered.write_bytes(canonical_json_bytes(changed) + b"\n")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    other = write_config(tmp_path / "other.json", make_config_mapping(seed=78))

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(output), "--config", str(other)]) == (
        EXIT_REJECTED
    )
    errors = capsys.readouterr().err
    assert "digest does not match" in errors
    assert "not canonical JSON" in errors
    assert "supplied config" in errors
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID


def test_run_refuses_inputs_and_outputs_before_it_measures(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    backend = StandInBackend()

    assert stand_in_main(run_argv(config, ceilings, config), backend) == EXIT_INVALID
    assert stand_in_main(run_argv(config, ceilings, ceilings), backend) == EXIT_INVALID
    assert stand_in_main(run_argv(config, ceilings, tmp_path), backend) == EXIT_INVALID
    broken = write_config(tmp_path / "broken.json", make_config_mapping(seed=-1))
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(broken, ceilings, output), backend) == EXIT_INVALID
    missing = tmp_path / "missing.json"
    assert stand_in_main(run_argv(config, missing, output), backend) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "must not alias the config" in errors
    assert "must not alias the ceiling report" in errors
    assert "ceiling report was refused" in errors
    assert backend.runs == [] and not output.exists()


def test_a_host_without_torch_exits_3_and_writes_nothing(tmp_path):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "new" / "report.json"

    completed = without_torch(*run_argv(PINNED, ceilings, output))

    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("metal-fused run unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_runs_without_torch(tmp_path, capsys):
    config, ceilings = inputs(tmp_path)
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, ceilings, output)) == EXIT_OK
    summary = capsys.readouterr().out.splitlines()[-1]

    completed = without_torch(
        "verify", "--report", str(output), "--ceilings", str(ceilings)
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert completed.stdout == f"verified_{summary}\n"
