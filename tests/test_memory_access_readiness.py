import json
import re
import shlex
import sys
from pathlib import Path

from ai_journey.memory_access_report_io import load_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/memory_access_*.py"
TESTS = "tests/test_memory_access_*.py"
PLAN = "config/day-74-memory-access-plan.json"
SMOKE = "config/day-74-memory-access-ci-smoke.json"


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def benchmark_flops(plan):
    benchmark = plan.benchmark
    work = sum(case.shape.flops for case in benchmark.cases())
    return work * (benchmark.warmup + benchmark.repeats)


def test_readiness_requires_every_memory_access_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {PLAN, SMOKE, "docs/memory-access-model.md"}

    assert "src/ai_journey/memory_access_cli.py" in expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]


def test_ci_smoke_plan_only_shrinks_the_pinned_cpu_benchmark():
    pinned = json.loads((ROOT / PLAN).read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / SMOKE).read_text(encoding="utf-8"))
    pinned_plan, smoke_plan = load_plan(ROOT / PLAN), load_plan(ROOT / SMOKE)

    assert smoke.pop("name") == "day-74-memory-access-ci-smoke"
    assert smoke.pop("benchmark")["backends"] == pinned.pop("benchmark")["backends"]
    assert pinned.pop("name") == "day-74-memory-access"
    assert smoke == pinned
    shapes = [shape.to_mapping() for shape in smoke_plan.benchmark.shapes]
    assert shapes == [{"m": 32, "n": 32, "k": 32}]
    assert benchmark_flops(smoke_plan) * 1000 < benchmark_flops(pinned_plan)


def test_ci_gates_the_lab_and_operates_the_installed_command():
    job = ci_job("memory-access-cpu")

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
        f'ai-journey-memory-access analyze --config "$GITHUB_WORKSPACE/{PLAN}"',
        f'--config "$GITHUB_WORKSPACE/{PLAN}"',
        f'ai-journey-memory-access benchmark --config "$GITHUB_WORKSPACE/{SMOKE}"',
        f'--config "$GITHUB_WORKSPACE/{SMOKE}"',
        "name: memory-access-evidence",
    ):
        assert command in job, command
    operations = [
        shlex.split(line)
        for line in job.splitlines()
        if line.strip().startswith("ai-journey-memory-access ")
    ]
    assert [words[1] for words in operations] == [
        "analyze",
        "verify",
        "benchmark",
        "verify",
    ]
    assert all(
        words[words.index("--config") + 1].endswith(SMOKE) for words in operations[2:]
    )
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern
