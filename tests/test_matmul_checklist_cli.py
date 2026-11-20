import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_checklist_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
    short_name,
)
from ai_journey.matmul_checklist_io import load_report, save_report
from ai_journey.matmul_checklist_source import MatmulChecklistError
from ai_journey.matmul_checklist_steps import CHECKLIST
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.matmul_checklist_verify import MatmulChecklistVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_correctness import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-77-matmul-checklist.json"
LOAD = (1.5, 2.25, 3.0)
NUMBER = r"\d+(?:\.\d+)?(?:e[-+]\d+)?"
RATE = r"\d+\.\d"
RATIO = r"(?:\d+\.\d{3}:\w+|none)"
SHORTS = [short_name(name) for name in VARIANTS]
CASE = r"correctness m=\d+ n=\d+ k=\d+ bound={0} ".format(NUMBER) + " ".join(
    rf"{short}_error={NUMBER} {short}_replay_bits=true" for short in SHORTS
)
POINT = r"benchmark m=\d+ n=\d+ k=\d+ " + " ".join(
    rf"{short}_gflops={RATE} {short}_iqr={RATE}-{RATE}" for short in (*SHORTS, "torch")
)
SPEEDUP = (
    rf"speedup m=\d+ n=\d+ k=\d+ step=S\d kernel=\w+ gflops=(?:{RATE}|none) "
    rf"to_base={RATIO} to_s0={RATIO} torch_fraction=(?:\d+\.\d{{3}}|none) "
    r"held=(?:true|false|none)"
)
BEST = (
    rf"best m=\d+ n=\d+ k=\d+ kernel=\w+ to_s0={RATIO} torch_fraction=\d+\.\d{{3}} "
    rf"torch_gflops={RATE} cublas=not_measured"
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_checklist_cli import main
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

    assert parser.prog == "ai-journey-matmul-checklist"
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
    assert [short_name(name) for name in ("matmul_s5_pad4", "torch_matmul")] == [
        "s5_pad4",
        "torch",
    ]


def test_a_stand_in_run_prints_the_checklist_and_every_number(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    label = "synthetic_timing"
    steps, rows = len(CHECKLIST), sum(len(step.variants) for step in CHECKLIST)
    assert len(lines) == steps + 1 + 3 + 2 + 2 * (rows + 1) + 1 + 1
    assert lines[0] == (
        f"{label} checklist step=S0 item=coalescing status=already_applied "
        "base=x_rows new=x_columns:faster"
    )
    assert lines[6] == (
        f"{label} checklist step=S6 item=occupancy status=new_kernel base=s5_pad4 "
        "new=s6_group_8x8:slower new=s6_group_32x32:no_clear_difference|slower"
    )
    assert lines[steps] == (
        f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    for line in lines[steps + 1 : steps + 4]:
        assert re.fullmatch(f"{label} {CASE}", line), line
    for line in lines[steps + 4 : steps + 6]:
        assert re.fullmatch(f"{label} {POINT}", line), line
    speedups = lines[steps + 6 : -2]
    assert [re.fullmatch(f"{label} {BEST}", line) is not None for line in speedups] == (
        [False] * rows + [True]
    ) * 2
    for line in speedups:
        assert re.fullmatch(f"{label} ({SPEEDUP}|{BEST})", line), line
    assert re.fullmatch(
        f"{label} warnings count=0 distinct=0 torch_imported_before_capture=(true|false)",
        lines[-2],
    )
    assert lines[-1] == (
        f"matmul_checklist_synthetic_timing={report['report_sha256']} "
        "correctness_cases=3 benchmark_points=2 refused=none hypotheses_held=8/18 "
        f"checklist={report['checklist']['sha256'][:12]} "
        "scope=synthetic_backend_or_clock load_before=1.50,2.25,3.00 "
        "load_after=1.50,2.25,3.00 measured=false cuda=false nvidia=false "
        "cublas=false hardware_counters=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_a_refused_kernel_is_printed_and_named_in_the_summary(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    backend = StandInBackend(threads={"matmul_s6_group_32x32": 512})

    assert stand_in_main(run_argv(config, output), backend=backend) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert lines[len(CHECKLIST) + 1] == (
        "synthetic_timing refused variant=matmul_s6_group_32x32 "
        "reason=pipeline_allows_fewer_threads_than_the_tile allowed_threads=512 "
        "static_threadgroup_bytes=18688"
    )
    assert "s6_group_32x32_error" not in "\n".join(lines)
    assert any(
        "kernel=s6_group_32x32 gflops=none to_base=none" in line for line in lines
    )
    assert " refused=matmul_s6_group_32x32 " in lines[-1]


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
    assert completed.stderr.startswith("matmul-checklist run unavailable: ")
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
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 78})

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    assert (
        main(["verify", "--report", str(output), "--config", str(other)])
        == EXIT_REJECTED
    )
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    with pytest.raises(MatmulChecklistVerificationError, match="digest"):
        load_report(tampered)
    with pytest.raises(MatmulChecklistError, match="non-symlink"):
        load_report(tmp_path)


def test_run_refuses_an_aliasing_output_a_device_error_and_a_spent_budget(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()
    assert stand_in_main(run_argv(config, config)) == EXIT_INVALID
    assert config.read_bytes() == before

    class FailingBackend(StandInBackend):
        def run(self, name, left, right, launch):
            raise RuntimeError("MPS backend out of memory")

    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, output), backend=FailingBackend()) == 2
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    short = write_config(tmp_path / "short.json", mapping)
    clock = scripted_clock(step=10**9)
    assert stand_in_main(run_argv(short, output), clock=clock) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "out of memory" in errors and "exceeded max_runtime_seconds=1" in errors
    assert not output.exists()


def test_save_report_verifies_before_it_writes(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    assert stand_in_main(run_argv(config, output)) == EXIT_OK
    report = json.loads(output.read_text(encoding="utf-8"))
    target = tmp_path / "copy.json"

    with pytest.raises(MatmulChecklistVerificationError):
        save_report(target, {**report, "measured": True})
    assert not target.exists()
    with pytest.raises(MatmulChecklistError, match="alias a protected input"):
        save_report(config, report, protected=(config,))
