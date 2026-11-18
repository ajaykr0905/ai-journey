import re
import shlex
from pathlib import Path

from ai_journey.matmul_model_cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
JOB = "matmul-model-cpu"
SOURCES = "src/ai_journey/matmul_model_*.py"
TESTS = "tests/test_matmul_model_*.py"
PINNED = "config/day-74-matmul-model.json"
EVIDENCE = "$RUNNER_TEMP/matmul-model-evidence/"


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def test_ci_gates_the_lab_without_torch_numpy_or_matplotlib():
    job = ci_job(JOB)

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
        "${{ runner.temp }}/matmul-model-evidence/",
        "name: matmul-model-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


def test_ci_runs_and_verifies_the_pinned_plan_with_the_installed_command():
    operations = [
        shlex.split(line)
        for line in ci_job(JOB).splitlines()
        if line.strip().startswith("ai-journey-matmul-model ")
    ]

    assert [words[1] for words in operations] == ["run", "verify"]
    run, verify = (build_parser().parse_args(words[1:]) for words in operations)
    assert run.config == verify.config == Path(f"$GITHUB_WORKSPACE/{PINNED}")
    assert verify.report == run.output
    assert str(run.output).startswith(EVIDENCE)
