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
from ai_journey.matmul_checklist_bound_io import load_bound_report
from ai_journey.matmul_checklist_bound_timing_io import load_timing_report
from ai_journey.matmul_checklist_cli import short_name
from ai_journey.matmul_checklist_steps import CHECKLIST
from ai_journey.matmul_checklist_variants import VARIANTS
from ai_journey.memory_access_plan import canonical_json_bytes
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_matmul_checklist_bound import timing_report
from test_matmul_checklist_bound_timing import TimingStandIn
from test_matmul_checklist_config import make_config_mapping
from test_matmul_checklist_recorded import RECORDED_CONFIG
from test_matmul_checklist_recorded import RECORDED_REPORT_JSON as CHECKLIST_JSON

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-77-matmul-checklist.json"
RULE = ROOT / "config/day-75-matmul-bottleneck.json"
RATE = r"\d+\.\d"
SHARE = r"\d+\.\d{3}"
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


def write_sources(tmp_path, timing=None):
    """Write the recorded checklist and ceiling reports and a timing report."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, report in (
        ("checklist", json.loads(CHECKLIST_JSON)),
        ("ceilings", json.loads(CEILING_JSON)),
        ("timing", timing or timing_report()),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_bytes(canonical_json_bytes(report) + b"\n")
    return paths


def source_argv(paths):
    return [
        "--checklist-report",
        str(paths["checklist"]),
        "--ceiling-report",
        str(paths["ceilings"]),
        "--timing-report",
        str(paths["timing"]),
        "--rule-config",
        str(RULE),
    ]


def test_bound_prints_each_side_and_step_and_verify_rebuilds_it(tmp_path, capsys):
    paths = write_sources(tmp_path)
    output = tmp_path / "bound.json"

    assert main(["bound", *source_argv(paths), "--output", str(output)]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, summary = load_bound_report(output)

    label = "synthetic_timing"
    sides, rows = len(VARIANTS) + 1, sum(len(step.variants) for step in CHECKLIST)
    assert len(lines) == 1 + 3 + 2 * (1 + sides) + 2 * rows + sides + 1
    assert re.fullmatch(
        rf"{label} ceilings fma_gflops=\d+\.\d bandwidth_gb_per_second=\d+\.\d\d "
        r"ridge_flops_per_byte=\d+\.\d\d fma_scales_with_work=false",
        lines[0],
    )
    assert lines[1].startswith(f"{label} source name=matmul_checklist report=")
    assert lines[4].startswith(f"{label} roof m=256 n=256 k=256 ")
    assert re.fullmatch(
        rf"{label} bound m=256 n=256 k=256 side=s7_interior gflops={RATE} "
        rf"fma_share={SHARE} model=tile_64x64x16_thread_4x4 "
        rf"flops_per_byte=\d+\.\d{{6}} attainable_gflops={RATE} "
        rf"attainable_share={SHARE} device_gb_per_second={RATE} "
        rf'threadgroup_reads_per_ns={RATE} name="[a-z -]+" reason=\w+',
        lines[14],
    )
    assert " side=x_columns " in lines[5] and " model=no_reuse " in lines[5]
    assert "threadgroup_reads_per_ns=none" in lines[5]
    assert " model=none " in lines[15] and " side=torch " in lines[15]
    steps = [line for line in lines if " step step=" in line]
    assert len(steps) == 2 * rows
    assert steps[0].startswith(
        f"{label} step step=S0 m=256 n=256 k=256 kernel=x_columns base=x_rows "
        "device_bytes_ratio=1.000000 load_operations_ratio=1.000000 "
        "threadgroup_reads_per_fma=none->none to_base="
    )
    assert "threadgroup_reads_per_fma=2.000000->1.250000" in steps[4]
    assert lines[-2].startswith(f'{label} top side=torch name="')
    assert lines[-1] == (
        f"matmul_checklist_bound_synthetic_timing={report['report_sha256']} points=2 "
        "sources_checked=true scope=synthetic_backend_or_clock measured=false "
        "counters=false nsight=false instruments=false cuda=false nvidia=false "
        "cublas=false"
    )
    assert summary["sources_checked"] is False
    argv = ["verify", "--report", str(output), *source_argv(paths)]
    assert main(argv) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
    assert main(["verify", "--report", str(output)]) == EXIT_OK
    assert "sources_checked=false" in capsys.readouterr().out


def test_verify_takes_all_three_sources_with_the_rule_config(tmp_path, capsys):
    paths = write_sources(tmp_path)
    report = tmp_path / "bound.json"
    partial = ["verify", "--report", str(report), "--checklist-report", "c.json"]
    sources = source_argv(paths)[:-2]

    for argv in (partial, ["verify", "--report", str(report), *sources]):
        with pytest.raises(SystemExit) as stopped:
            main(argv)
        assert stopped.value.code == EXIT_INVALID
    assert "--rule-config" in capsys.readouterr().err


def test_bound_rejects_tampered_sources_and_refuses_unpaired_ones(tmp_path, capsys):
    paths = write_sources(tmp_path)
    output = tmp_path / "bound.json"
    source = json.loads(paths["checklist"].read_text())
    source["label"] = "synthetic timing, not a measurement"
    paths["checklist"].write_bytes(canonical_json_bytes(source) + b"\n")

    assert main(["bound", *source_argv(paths), "--output", str(output)]) == (
        EXIT_REJECTED
    )
    assert "rejected: checklist report" in capsys.readouterr().err
    other = write_sources(
        tmp_path, timing_report(mapping={**RECORDED_CONFIG, "seed": 78})
    )
    assert main(["bound", *source_argv(other), "--output", str(output)]) == (
        EXIT_INVALID
    )
    assert "checklist report's config" in capsys.readouterr().err
    assert not output.exists()


def test_verify_rejects_a_tampered_bound_and_a_rebuild_that_differs(tmp_path, capsys):
    paths = write_sources(tmp_path)
    output = tmp_path / "bound.json"
    assert main(["bound", *source_argv(paths), "--output", str(output)]) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text())

    tampered = tmp_path / "tampered.json"
    report["points"][0]["calls_per_sample"] = 3
    tampered.write_bytes(canonical_json_bytes(report) + b"\n")
    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert "digest does not match" in capsys.readouterr().err
    other = write_sources(tmp_path / "other", timing_report(scale={"matmul_x_rows": 2}))
    argv = ["verify", "--report", str(output), *source_argv(other)]
    assert main(argv) == EXIT_REJECTED
    assert "supplied source reports" in capsys.readouterr().err


def test_bound_refuses_an_output_that_aliases_an_input(tmp_path, capsys):
    paths = write_sources(tmp_path)
    before = paths["timing"].read_bytes()

    argv = ["bound", *source_argv(paths), "--output", str(paths["timing"])]
    assert main(argv) == EXIT_INVALID
    assert "must not alias an input" in capsys.readouterr().err
    assert paths["timing"].read_bytes() == before
