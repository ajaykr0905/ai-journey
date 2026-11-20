import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_checklist_bound_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.matmul_checklist_bound_timing_io import load_timing_report
from ai_journey.matmul_checklist_cli import short_name
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_checklist_bound_timing import TimingStandIn
from test_matmul_checklist_config import make_config_mapping

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-77-matmul-checklist.json"
LOAD = (1.5, 2.25, 3.0)
SHORTS = [short_name(name) for name in (*VARIANTS, "torch_matmul")]
ONE_CALL = r"one_call m=\d+ n=\d+ k=\d+ " + " ".join(
    rf"{short}_gpu_ns=\d+ {short}_host_ns=\d+ {short}_dispatch_ns=\d+"
    for short in SHORTS
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_checklist_bound_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(path, mapping=None):
    path.write_text(json.dumps(mapping or make_config_mapping()), encoding="utf-8")
    return path


def scripted_clock(step=1_000):
    return itertools.count(0, step).__next__


def stand_in_main(argv, backend=None):
    return main(
        argv,
        backend=backend or TimingStandIn(),
        clock=scripted_clock(),
        load_average=lambda: LOAD,
    )


def timing_argv(config, output):
    return ["timing", "--config", str(config), "--output", str(output)]


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

    assert parser.prog == "ai-journey-matmul-checklist-bound"
    parsed = parser.parse_args(timing_argv("c.json", "t.json"))
    assert (parsed.command, str(parsed.config), str(parsed.output)) == (
        "timing",
        "c.json",
        "t.json",
    )
    assert parser.parse_args(["verify-timing", "--report", "t.json"]).config is None
    for argv in ([], ["timing", "--config", "c.json"], ["timing", "--conf", "c.json"]):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(argv)
        assert stopped.value.code == EXIT_INVALID
    assert (EXIT_OK, EXIT_REJECTED, EXIT_INVALID, EXIT_UNAVAILABLE) == (0, 1, 2, 3)


def test_a_stand_in_timing_prints_every_one_call_median(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "timing.json"

    assert stand_in_main(timing_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_timing_report(output)

    label = "synthetic_timing"
    assert len(lines) == 1 + 3 + 1 + 1
    assert lines[0] == (
        f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    for line in lines[1:4]:
        assert re.fullmatch(f"{label} {ONE_CALL}", line), line
    assert lines[1].startswith(f"{label} one_call m=1 n=1 k=1 x_columns_gpu_ns=500 ")
    assert lines[4] == (
        f"{label} warnings count=0 distinct=0 torch_imported_before_capture="
        f"{str(report['warnings']['torch_imported_before_capture']).lower()}"
    )
    assert lines[5] == (
        f"matmul_checklist_one_call_synthetic_timing={report['report_sha256']} "
        f"points=2 sides={','.join(SHORTS)} refused=none warnings=0 "
        "scope=synthetic_backend_or_clock load_before=1.50,2.25,3.00 "
        "load_after=1.50,2.25,3.00 measured=false cuda=false nvidia=false "
        "cublas=false hardware_counters=false"
    )
    argv = ["verify-timing", "--report", str(output), "--config", str(config)]
    assert main(argv) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[5]}\n"


def test_a_refused_kernel_is_printed_and_named_in_the_summary(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    backend = TimingStandIn(threads={"matmul_s6_group_32x32": 512})

    argv = timing_argv(config, tmp_path / "timing.json")
    assert stand_in_main(argv, backend=backend) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert lines[1] == (
        "synthetic_timing refused variant=matmul_s6_group_32x32 "
        "reason=pipeline_allows_fewer_threads_than_the_tile allowed_threads=512 "
        "static_threadgroup_bytes=18688"
    )
    assert "s6_group_32x32_gpu_ns" not in "\n".join(lines)
    assert " refused=matmul_s6_group_32x32 " in lines[-1]


def test_a_host_without_torch_exits_3_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "timing.json"

    completed = without_torch(*timing_argv(PINNED, output))

    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr.startswith("matmul-checklist-bound timing unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_timing_rejects_tampered_or_reformatted_reports(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "timing.json"
    assert stand_in_main(timing_argv(config, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    report["points"][0]["agreement_bound"] = 0.5
    tampered.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify-timing", "--report", str(tampered)]) == EXIT_REJECTED
    assert "digest does not match" in capsys.readouterr().err
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(json.loads(output.read_text()), indent=2))
    assert main(["verify-timing", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical JSON" in capsys.readouterr().err
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 78})
    argv = ["verify-timing", "--report", str(output), "--config", str(other)]
    assert main(argv) == EXIT_REJECTED
    assert "supplied config" in capsys.readouterr().err
    missing = ["verify-timing", "--report", str(tmp_path / "missing.json")]
    assert main(missing) == EXIT_INVALID
    assert capsys.readouterr().err.startswith("matmul-checklist-bound verify-timing")


def test_timing_refuses_an_output_that_aliases_its_config(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()

    assert stand_in_main(timing_argv(config, config)) == EXIT_INVALID
    assert "must not alias the config" in capsys.readouterr().err
    assert config.read_bytes() == before
