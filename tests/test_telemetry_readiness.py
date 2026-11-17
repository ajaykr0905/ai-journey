import json
import re
import shlex
import sys
from pathlib import Path

from ai_journey import (
    recovery_cli,
    serving_recovery_cli,
    telemetry_metrics_cli,
    telemetry_traces_cli,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/telemetry_*.py"
TESTS = "tests/test_telemetry_*.py"
METRICS_CONFIG = "config/day-73-telemetry-metrics.json"
TRACES_CONFIG = "config/day-73-telemetry-traces.json"
GUIDES = ("docs/telemetry-metrics.md", "docs/telemetry-traces-dashboards.md")
EVIDENCE = "$RUNNER_TEMP/telemetry-evidence/"
PARSERS = {
    "ai-journey-serving-recovery": serving_recovery_cli._parser,
    "ai-journey-recovery": recovery_cli._parser,
    "ai-journey-telemetry-metrics": telemetry_metrics_cli.build_parser,
    "ai-journey-telemetry-traces": telemetry_traces_cli.build_parser,
}


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def checkout(path):
    return Path(f"$GITHUB_WORKSPACE/{path}")


def test_readiness_requires_every_telemetry_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {METRICS_CONFIG, TRACES_CONFIG, *GUIDES}

    assert {
        "src/ai_journey/telemetry_metrics_cli.py",
        "src/ai_journey/telemetry_traces_cli.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]


def test_ci_gates_both_labs_and_installs_no_torch_numpy_or_matplotlib():
    job = ci_job("telemetry-cpu")

    assert 'python-version: "3.11"' in job
    installs = [line.lower() for line in job.splitlines() if "pip install" in line]
    assert installs, "the job must install the package and its tools"
    for dependency in ("torch", "numpy", "matplotlib"):
        assert not [line for line in installs if dependency in line], dependency
    assert "python -m pip install -e . --no-deps" in job
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f"python -m pytest -q {TESTS}",
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "${{ runner.temp }}/telemetry-evidence/",
        "name: telemetry-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


def test_ci_feeds_fresh_recovery_runs_through_every_installed_telemetry_command():
    operations = [
        shlex.split(line)
        for line in ci_job("telemetry-cpu").splitlines()
        if line.strip().startswith("ai-journey-")
    ]

    assert [words[:2] for words in operations] == [
        ["ai-journey-serving-recovery", "run"],
        ["ai-journey-recovery", "run"],
        ["ai-journey-telemetry-metrics", "export"],
        ["ai-journey-telemetry-metrics", "verify"],
        ["ai-journey-telemetry-traces", "export"],
        ["ai-journey-telemetry-traces", "dashboard"],
        ["ai-journey-telemetry-traces", "verify"],
        ["ai-journey-telemetry-traces", "verify"],
    ]
    (
        serving,
        worker,
        metrics,
        metrics_verify,
        traces,
        dashboard,
        traces_verify,
        dashboard_verify,
    ) = (PARSERS[words[0]]().parse_args(words[1:]) for words in operations)
    assert serving.source == worker.source == checkout("data/day-19-demo-names.txt")
    assert serving.config == checkout("config/day-70-serving-recovery.json")
    assert worker.config == checkout("config/day-70-worker-recovery.json")
    runs = (serving.run_dir, worker.run_dir)
    for command in (metrics, metrics_verify, traces, traces_verify):
        assert (command.serving_run_dir, command.worker_run_dir) == runs
    assert metrics.config == metrics_verify.config == checkout(METRICS_CONFIG)
    assert metrics_verify.output_dir == metrics.output_dir
    for command in (traces, dashboard, traces_verify, dashboard_verify):
        assert command.config == checkout(TRACES_CONFIG)
    assert traces_verify.report == traces.output_dir / "report.json"
    assert dashboard_verify.report == dashboard.output_dir / "report.json"
    assert dashboard_verify.serving_run_dir is None
    assert dashboard_verify.worker_run_dir is None
    written = (*runs, metrics.output_dir, traces.output_dir, dashboard.output_dir)
    assert len(set(written)) == len(written)
    assert all(str(path).startswith(EVIDENCE) for path in written)
