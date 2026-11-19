import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.matmul_tiled_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    EXIT_UNAVAILABLE,
    build_parser,
    main,
    short_name,
)
from ai_journey.matmul_tiled_io import load_report, save_report
from ai_journey.matmul_tiled_source import MatmulTiledError
from ai_journey.matmul_tiled_verify import MatmulTiledVerificationError
from ai_journey.memory_access_plan import canonical_json_bytes
from test_matmul_tiled_config import make_config_mapping
from test_matmul_tiled_correctness import StandInBackend

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-76-matmul-tiled.json"
LOAD = (1.5, 2.25, 3.0)
NUMBER = r"\d+(?:\.\d+)?(?:e[-+]\d+)?"
RATE = r"\d+\.\d"
SHORTS = ("naive", "tiled_8", "tiled_16", "tiled_32")
CASE = r"correctness m=\d+ n=\d+ k=\d+ bound={0} ".format(NUMBER) + " ".join(
    rf"{short}_error={NUMBER} {short}_replay_bits=true" for short in SHORTS
)
POINT = r"benchmark m=\d+ n=\d+ k=\d+ " + " ".join(
    rf"{short}_gflops={RATE} {short}_iqr={RATE}-{RATE}" for short in (*SHORTS, "torch")
)
NUMBERS = (
    rf"three_numbers m=\d+ n=\d+ k=\d+ naive_gflops={RATE} tiled=tiled_\d+:{RATE} "
    rf"vendor_gflops={RATE} cublas=not_measured "
    r"tiled_over_naive=\d+\.\d{3}:\w+ vendor_over_tiled=\d+\.\d{3}:\w+"
)
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.matmul_tiled_cli import main
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

    assert parser.prog == "ai-journey-matmul-tiled"
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
    assert [short_name(name) for name in ("matmul_tiled_16", "torch_matmul")] == [
        "tiled_16",
        "torch",
    ]


def test_a_stand_in_run_prints_every_number_and_verifies_as_synthetic(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"

    assert stand_in_main(run_argv(config, output)) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(output)

    label = "synthetic_timing"
    assert len(lines) == 1 + 3 + 2 + 2 + 1 + 1
    assert (
        lines[0] == f"{label} load_average before=1.50,2.25,3.00 after=1.50,2.25,3.00"
    )
    for line in lines[1:4]:
        assert re.fullmatch(f"{label} {CASE}", line), line
    for line in lines[4:6]:
        assert re.fullmatch(f"{label} {POINT}", line), line
    for line in lines[6:8]:
        assert re.fullmatch(f"{label} {NUMBERS}", line), line
    assert re.fullmatch(
        f"{label} warnings count=0 distinct=0 torch_imported_before_capture=(true|false)",
        lines[8],
    )
    assert lines[-1] == (
        f"matmul_tiled_synthetic_timing={report['report_sha256']} "
        "correctness_cases=3 benchmark_points=2 refused=none "
        "scope=synthetic_backend_or_clock load_before=1.50,2.25,3.00 "
        "load_after=1.50,2.25,3.00 measured=false cuda=false nvidia=false "
        "cublas=false"
    )
    assert main(["verify", "--report", str(output), "--config", str(config)]) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_a_refused_tile_is_printed_and_named_in_the_summary(tmp_path, capsys):
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "report.json"
    backend = StandInBackend(threads={"matmul_tiled_32": 512})

    assert stand_in_main(run_argv(config, output), backend=backend) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()

    assert lines[1] == (
        "synthetic_timing refused variant=matmul_tiled_32 "
        "reason=pipeline_allows_fewer_threads_than_the_tile allowed_threads=512 "
        "static_threadgroup_bytes=8192"
    )
    assert "tiled_32_error" not in "\n".join(lines)
    assert " refused=matmul_tiled_32 " in lines[-1]


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
    assert completed.stderr.startswith("matmul-tiled run unavailable: ")
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
    other = write_config(tmp_path / "other.json", {**make_config_mapping(), "seed": 77})

    assert main(["verify", "--report", str(tampered)]) == EXIT_REJECTED
    assert main(["verify", "--report", str(pretty)]) == EXIT_REJECTED
    assert "not canonical" in capsys.readouterr().err
    assert (
        main(["verify", "--report", str(output), "--config", str(other)])
        == EXIT_REJECTED
    )
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == EXIT_INVALID
    with pytest.raises(MatmulTiledVerificationError, match="digest"):
        load_report(tampered)
    with pytest.raises(MatmulTiledError, match="non-symlink"):
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

    with pytest.raises(MatmulTiledVerificationError):
        save_report(target, {**report, "measured": True})
    assert not target.exists()
    with pytest.raises(MatmulTiledError, match="alias a protected input"):
        save_report(config, report, protected=(config,))
