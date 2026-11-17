import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey.kernel_launch_io import canonical_json_bytes
from ai_journey.matmul_model_cli import EXIT_INVALID, EXIT_OK, EXIT_REJECTED, main
from ai_journey.matmul_model_report import DIGEST_FIELD, seal_report
from ai_journey.matmul_model_verify import load_report
from test_matmul_model_plan import plan_mapping, write_plan

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "config/day-74-matmul-model.json"
MODULES = (
    "ai_journey.matmul_model_access",
    "ai_journey.matmul_model_cli",
    "ai_journey.matmul_model_crosscheck",
    "ai_journey.matmul_model_cuda",
    "ai_journey.matmul_model_kernels",
    "ai_journey.matmul_model_occupancy",
    "ai_journey.matmul_model_plan",
    "ai_journey.matmul_model_reference",
    "ai_journey.matmul_model_report",
    "ai_journey.matmul_model_roofline",
    "ai_journey.matmul_model_trace",
    "ai_journey.matmul_model_traffic",
    "ai_journey.matmul_model_verify",
)
NEWLY_IMPORTED = """
import json, sys
before = set(sys.modules)
import {module}
print(json.dumps(sorted(set(sys.modules) - before)))
"""
# A None entry in sys.modules makes every import of that name fail, as on a
# CI runner that installs none of these packages.
WITHOUT_HEAVY_PACKAGES = """
import sys
for name in ("torch", "numpy", "matplotlib"):
    sys.modules[name] = None
from ai_journey.matmul_model_cli import main
sys.exit(main(sys.argv[1:]))
"""


def run_fresh(*arguments, cwd=None):
    return subprocess.run(
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_run_then_verify_print_one_digest_line(tmp_path: Path, capsys):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "out" / "report.json"

    assert main(["run", "--config", str(config), "--output", str(output)]) == EXIT_OK
    built = capsys.readouterr().out.strip()
    report = load_report(output)
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    summary = report["summary"]
    assert built == (
        f"matmul_model_report={report[DIGEST_FIELD]} runs=8 "
        "runs_matching_reference=8 order_independent_launches=4/4 "
        f"cross_checked_requests={summary['cross_checked_requests']} "
        "probe_traces=8/8 memory_bound_points=5/6 cuda_checks=16/16 "
        "simulated_on_cpu=true gpu_executed=false cuda_compiled=false"
    )

    for arguments in (
        ["verify", "--report", str(output)],
        ["verify", "--report", str(output), "--config", str(config)],
    ):
        assert main(arguments) == EXIT_OK
        verified = capsys.readouterr().out.strip()
        assert verified == built.replace(
            "matmul_model_report=", "verified_matmul_model_report=", 1
        )


def test_run_never_writes_over_its_own_plan(tmp_path: Path, capsys):
    config = write_plan(tmp_path / "plan.json")
    before = config.read_bytes()
    hard = tmp_path / "hard.json"
    os.link(config, hard)
    soft = tmp_path / "soft.json"
    soft.symlink_to(config)

    for target in (config, hard, soft):
        code = main(["run", "--config", str(config), "--output", str(target)])
        assert code == EXIT_INVALID
        assert "must not alias" in capsys.readouterr().err
    assert config.read_bytes() == before


def test_unusable_outputs_return_the_invalid_exit_code(tmp_path: Path, capsys):
    config = write_plan(tmp_path / "plan.json")
    loop = tmp_path / "loop"
    loop.symlink_to("loop")

    for output in (".", str(tmp_path / ".."), str(loop), str(loop / "report.json")):
        code = main(["run", "--config", str(config), "--output", output])
        assert code == EXIT_INVALID, output
        assert "matmul-model run failed" in capsys.readouterr().err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["loop", "plan.json"]


def test_invalid_plans_fail_and_rejected_reports_are_named(tmp_path: Path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_bytes(b'{"name":"a","name":"b"}')
    assert main(["run", "--config", str(config), "--output", str(output)]) == 2
    assert "duplicate JSON key: name" in capsys.readouterr().err
    for key, value in (
        ("block", [64, 32]),
        ("block", [16, 16]),
        ("registers_per_thread", [True]),
    ):
        write_plan(config, {**plan_mapping(), key: value})
        assert main(["run", "--config", str(config), "--output", str(output)]) == 2
        assert "matmul-model run failed" in capsys.readouterr().err
    assert not output.exists()

    write_plan(config)
    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    capsys.readouterr()
    report = load_report(output)
    edited = {**report, "summary": {**report["summary"], "runs": 9}}
    output.write_bytes(canonical_json_bytes(edited) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "digest does not match" in capsys.readouterr().err

    unsealed = {key: value for key, value in edited.items() if key != DIGEST_FIELD}
    output.write_bytes(canonical_json_bytes(seal_report(unsealed)) + b"\n")
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "summary.runs does not match" in capsys.readouterr().err

    other = write_plan(tmp_path / "other.json", {**plan_mapping(), "name": "other"})
    write_plan(config)
    main(["run", "--config", str(config), "--output", str(output)])
    capsys.readouterr()
    code = main(["verify", "--report", str(output), "--config", str(other)])
    assert code == EXIT_REJECTED
    assert "not built from this plan" in capsys.readouterr().err

    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == 2
    assert "readable regular non-symlink" in capsys.readouterr().err


def test_overlong_json_integers_are_refused_with_explicit_codes(tmp_path, capsys):
    # 5,000 digits exceed CPython's default 4,300-digit limit for int().
    huge = tmp_path / "huge.json"
    huge.write_bytes(b'{"schema_version":' + b"1" * 5_000 + b"}")
    output = tmp_path / "report.json"

    assert main(["run", "--config", str(huge), "--output", str(output)]) == 2
    assert main(["verify", "--report", str(huge)]) == EXIT_REJECTED
    assert capsys.readouterr().err.count("integer literal too long") == 2
    assert not output.exists()


def test_usage_errors_exit_with_the_argparse_status():
    for arguments in (
        ["run", "--config", "plan.json"],
        ["verify", "--rep", "report.json"],
        ["build", "--config", "plan.json"],
        [],
    ):
        with pytest.raises(SystemExit) as stopped:
            main(arguments)
        assert stopped.value.code == 2


def test_every_matmul_model_module_loads_neither_torch_nor_numpy():
    listed = {
        f"ai_journey.{path.stem}"
        for path in (ROOT / "src/ai_journey").glob("matmul_model_*.py")
    }
    assert set(MODULES) == listed
    for module in MODULES:
        completed = run_fresh("-c", NEWLY_IMPORTED.format(module=module))
        assert completed.returncode == 0, completed.stderr
        loaded = set(json.loads(completed.stdout))

        assert module in loaded
        assert {"torch", "numpy", "matplotlib"} & loaded == set(), module


def test_pinned_run_and_verify_work_without_torch_or_numpy(tmp_path: Path, capsys):
    expected = tmp_path / "expected.json"
    assert main(["run", "--config", str(PINNED), "--output", str(expected)]) == 0
    capsys.readouterr()
    output = tmp_path / "report.json"

    built = run_fresh(
        "-c",
        WITHOUT_HEAVY_PACKAGES,
        "run",
        "--config",
        str(PINNED),
        "--output",
        str(output),
    )
    verified = run_fresh(
        "-m",
        "ai_journey.matmul_model_cli",
        "verify",
        "--report",
        str(output),
        "--config",
        str(PINNED),
        cwd=tmp_path,
    )

    assert built.returncode == 0, built.stderr
    assert built.stdout.startswith("matmul_model_report=")
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.startswith("verified_matmul_model_report=")
    assert verified.stdout.split(" ", 1)[1] == built.stdout.split(" ", 1)[1]
    assert output.read_bytes() == expected.read_bytes()
