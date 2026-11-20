import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_tiled_gap_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
)
from ai_journey.matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
)
from ai_journey.matmul_tiled_gap_timing_io import (
    load_timing_report,
    save_timing_report,
)
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_tiled_config import make_config_mapping
from test_matmul_tiled_gap import ceilings, tiled, timing_report
from test_matmul_tiled_gap_timing import LOAD, TimingStandIn, scripted_clock
from test_matmul_tiled_recorded import RECORDED_CONFIG

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-76-matmul-tiled.json"
RULE_CONFIG = ROOT / "config/day-75-matmul-bottleneck.json"
GAP_LINE = (
    r"gap m=\d+ n=\d+ k=\d+ side=\w+ gflops=\d+\.\d fma_share=\S+ "
    r"model=(no_reuse|tiled_\d+|none) model_flops_per_byte=(\d+\.\d{6}|none) "
    r"attainable_gflops=(\d+\.\d|none) attainable_share=\S+ "
    r'ideal_share=\S+ name="[\w -]+" reason=\w+'
)
SHORTS = ("naive", "tiled_8", "tiled_16", "tiled_32", "torch")
ONE_CALL = r"one_call m=\d+ n=\d+ k=\d+ " + " ".join(
    rf"{short}_{field}=\d+"
    for short in SHORTS
    for field in ("gpu_ns", "host_ns", "dispatch_ns")
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_tiled_gap_cli import main
sys.exit(main(sys.argv[1:]))
"""


def write_config(path, mapping=None):
    path.write_text(json.dumps(mapping or make_config_mapping()), encoding="utf-8")
    return path


def stand_in_main(argv, backend=None, clock=None):
    return main(
        argv,
        backend=backend or TimingStandIn(),
        clock=clock or scripted_clock(),
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

    assert parser.prog == "ai-journey-matmul-tiled-gap"
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


def test_a_stand_in_timing_prints_each_median_and_verifies_as_synthetic(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "timing.json"

    assert stand_in_main(timing_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_timing_report(output)

    label = "synthetic_timing"
    assert len(lines) == 1 + 3 + 1
    assert (
        lines[0] == f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    for line in lines[1:4]:
        assert re.fullmatch(f"{label} {ONE_CALL}", line), line
    assert lines[1].startswith(f"{label} one_call m=1 n=1 k=1 naive_gpu_ns=500 ")
    assert lines[-1] == (
        f"matmul_tiled_one_call_synthetic_timing={report['report_sha256']} "
        "points=2 sides=naive,tiled_8,tiled_16,tiled_32,torch "
        "scope=synthetic_backend_or_clock load_before=1.50,2.25,3.00 "
        "load_after=1.50,2.25,3.00 measured=false cuda=false nvidia=false "
        "cublas=false"
    )
    verify = ["verify-timing", "--report", str(output), "--config", str(config)]
    assert main(verify) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


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
    assert completed.stderr.startswith("matmul-tiled-gap timing unavailable: ")
    assert f"({reason})" in completed.stderr
    assert not (tmp_path / "new").exists()


def test_verify_timing_rejects_tampered_reformatted_and_mismatched_reports(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "timing.json"
    assert stand_in_main(timing_argv(config, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))

    tampered = tmp_path / "tampered.json"
    changed = {**report, "label": "host-specific Apple GPU (Metal) measurement"}
    tampered.write_bytes(canonical_json_bytes(changed) + b"\n")
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 77})

    assert main(["verify-timing", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify-timing", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    mismatched = ["verify-timing", "--report", str(output), "--config", str(other)]
    assert main(mismatched) == EXIT_REJECTED
    missing = ["verify-timing", "--report", str(tmp_path / "missing.json")]
    assert main(missing) == EXIT_INVALID
    with pytest.raises(MatmulTiledGapVerificationError, match="digest"):
        load_timing_report(tampered)
    with pytest.raises(MatmulTiledGapError, match="non-symlink"):
        load_timing_report(tmp_path)


def test_timing_refuses_an_aliasing_output_a_device_error_and_a_spent_budget(
    tmp_path, capsys
):
    config = write_config(tmp_path / "config.json")
    before = config.read_bytes()
    assert stand_in_main(timing_argv(config, config)) == EXIT_INVALID
    assert config.read_bytes() == before

    class FailingBackend(TimingStandIn):
        def event_pair(self):
            raise RuntimeError("MPS backend out of memory")

    output = tmp_path / "timing.json"
    assert stand_in_main(timing_argv(config, output), FailingBackend()) == 2
    mapping = make_config_mapping()
    mapping["benchmark"]["max_runtime_seconds"] = 1
    short = write_config(tmp_path / "short.json", mapping)
    clock = scripted_clock(step=10**9)
    assert stand_in_main(timing_argv(short, output), clock=clock) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "out of memory" in errors and "exceeded max_runtime_seconds=1" in errors
    assert not output.exists()


def test_save_timing_report_verifies_before_it_writes(tmp_path):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "timing.json"
    assert stand_in_main(timing_argv(config, output)) == EXIT_OK
    report = json.loads(output.read_text(encoding="utf-8"))
    target = tmp_path / "copy.json"

    with pytest.raises(MatmulTiledGapVerificationError):
        save_timing_report(target, {**report, "measured": True})
    assert not target.exists()
    with pytest.raises(MatmulTiledGapError, match="alias a protected input"):
        save_timing_report(config, report, protected=(config,))


def write_canonical(path, report):
    path.write_bytes(canonical_json_bytes(report) + b"\n")
    return path


def gap_sources(directory, timing=None):
    directory.mkdir(exist_ok=True)
    return {
        "tiled_report": write_canonical(directory / "tiled.json", tiled()),
        "ceiling_report": write_canonical(directory / "ceilings.json", ceilings()),
        "timing_report": write_canonical(
            directory / "timing.json", timing or timing_report()
        ),
        "rule_config": RULE_CONFIG,
    }


def source_argv(sources):
    argv = []
    for name, path in sources.items():
        argv += [f"--{name.replace('_', '-')}", str(path)]
    return argv


def gap_argv(sources, output):
    return ["gap", *source_argv(sources), "--output", str(output)]


def test_gap_prints_every_side_and_verify_rebuilds_it_from_the_sources(
    tmp_path, capsys
):
    sources = gap_sources(tmp_path)
    output = tmp_path / "gap.json"

    assert main(gap_argv(sources, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(output.read_text(encoding="utf-8"))

    label = "synthetic_timing"
    assert len(lines) == 1 + 3 + 2 * (1 + 5) + 5 + 1
    assert lines[0].startswith(f"{label} ceilings fma_gflops=1802.7 ")
    assert lines[1].startswith(f"{label} source name=matmul_tiled report=")
    assert lines[4].startswith(f"{label} roof m=256 n=256 k=256 ")
    for line in lines[5:10] + lines[11:16]:
        assert re.fullmatch(f"{label} {GAP_LINE}", line), line
    assert "side=torch gflops=54.9 fma_share=0.0305 model=none " in lines[9]
    assert lines[16] == (
        f'{label} top side=naive name="latency or occupancy-bound" points=2/2 '
        "reason=most_points"
    )
    assert lines[-1] == (
        f"matmul_tiled_gap_synthetic_timing={report['report_sha256']} points=2 "
        "sources_checked=true scope=synthetic_backend_or_clock measured=false "
        "counters=false nsight=false instruments=false cuda=false nvidia=false "
        "cublas=false"
    )
    verify = ["verify", "--report", str(output)]
    assert main([*verify, *source_argv(sources)]) == EXIT_OK
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
    assert main(verify) == EXIT_OK
    assert "sources_checked=false" in capsys.readouterr().out


def test_verify_needs_every_source_and_the_rule_config_to_rebuild(tmp_path, capsys):
    sources = gap_sources(tmp_path)
    output = tmp_path / "gap.json"
    assert main(gap_argv(sources, output)) == EXIT_OK
    capsys.readouterr()
    partial = ["--tiled-report", str(sources["tiled_report"])]
    without_rule = {
        name: path for name, path in sources.items() if name != "rule_config"
    }

    for argv in (partial, source_argv(without_rule)):
        with pytest.raises(SystemExit) as stopped:
            main(["verify", "--report", str(output), *argv])
        assert stopped.value.code == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "together, or none of them" in errors
    assert "verify needs --rule-config" in errors


def test_gap_refuses_an_aliasing_output_unpaired_and_tampered_sources(tmp_path, capsys):
    sources = gap_sources(tmp_path)
    before = sources["timing_report"].read_bytes()
    assert main(gap_argv(sources, sources["timing_report"])) == EXIT_INVALID
    assert sources["timing_report"].read_bytes() == before

    output = tmp_path / "gap.json"
    other = timing_report(mapping={**RECORDED_CONFIG, "seed": 77})
    unpaired = gap_sources(tmp_path / "unpaired", other)
    assert main(gap_argv(unpaired, output)) == EXIT_INVALID
    report = json.loads(sources["tiled_report"].read_text(encoding="utf-8"))
    tampered = {
        **sources,
        "tiled_report": write_canonical(
            tmp_path / "tampered.json", {**report, "measured": False}
        ),
    }
    assert main(gap_argv(tampered, output)) == EXIT_REJECTED
    missing = {**sources, "ceiling_report": tmp_path / "missing.json"}
    assert main(gap_argv(missing, output)) == EXIT_INVALID
    errors = capsys.readouterr().err
    assert "must not alias an input" in errors
    assert "tiled report's config" in errors
    assert "rejected: tiled report:" in errors
    assert not output.exists()


def test_verify_rejects_a_tampered_or_reformatted_gap(tmp_path, capsys):
    sources = gap_sources(tmp_path)
    output = tmp_path / "gap.json"
    assert main(gap_argv(sources, output)) == EXIT_OK
    capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    tampered = write_canonical(tmp_path / "tampered.json", {**report, "points": []})
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "none.json")]) == EXIT_INVALID
