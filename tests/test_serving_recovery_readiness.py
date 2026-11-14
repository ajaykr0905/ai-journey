import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/serving_recovery_*.py"
TESTS = "tests/test_serving_recovery_*.py"


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def test_readiness_requires_every_serving_recovery_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {
        "config/day-70-serving-recovery.json",
        "docs/serving-failure-recovery.md",
    }

    assert "src/ai_journey/serving_recovery_cli.py" in expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]


def test_ci_gates_the_lab_and_runs_the_installed_command_outside_the_checkout():
    job = ci_job("serving-recovery-cpu")

    assert 'python-version: "3.11"' in job
    assert "python -m pip install torch --index-url" in job
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
        "ai-journey-serving-recovery run --source "
        '"$GITHUB_WORKSPACE/data/day-19-demo-names.txt" --config '
        '"$GITHUB_WORKSPACE/config/day-70-serving-recovery.json"',
        'ai-journey-serving-recovery verify --run-dir "$RUNNER_TEMP/',
        "name: serving-recovery-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern
