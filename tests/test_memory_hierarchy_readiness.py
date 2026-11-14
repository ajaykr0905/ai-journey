import json
import re
import shlex
import sys
from pathlib import Path

from ai_journey.memory_hierarchy_config import load_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/memory_hierarchy_*.py"
TESTS = "tests/test_memory_hierarchy_*.py"
PINNED = "config/day-71-memory-hierarchy.json"
SMOKE = "config/day-71-memory-hierarchy-ci-smoke.json"


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def test_readiness_requires_every_memory_hierarchy_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {PINNED, SMOKE, "docs/memory-hierarchy-lab.md"}

    assert "src/ai_journey/memory_hierarchy_cli.py" in expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]


def test_ci_smoke_config_only_shrinks_the_pinned_run():
    pinned = json.loads((ROOT / PINNED).read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / SMOKE).read_text(encoding="utf-8"))
    pinned_work = load_config(ROOT / PINNED).planned_work()
    smoke_work = load_config(ROOT / SMOKE).planned_work()

    assert pinned.pop("name") == "day-71-memory-hierarchy"
    assert smoke.pop("name") == "day-71-memory-hierarchy-ci-smoke"
    timing, smoke_timing = pinned.pop("timing"), smoke.pop("timing")
    assert smoke_timing["max_runtime_seconds"] == timing["max_runtime_seconds"]
    for field in ("warmup", "repeats", "target_bytes_per_sample"):
        assert smoke_timing[field] < timing[field], field
    stream, smoke_stream = pinned.pop("stream"), smoke.pop("stream")
    assert smoke_stream["kernels"] == stream["kernels"]
    assert set(smoke_stream["elements"]) < set(stream["elements"])
    access, smoke_access = pinned.pop("access"), smoke.pop("access")
    assert smoke_access["stride_elements"] == access["stride_elements"]
    assert smoke_access["elements"] < access["elements"]
    matmul, smoke_matmul = pinned.pop("matmul"), smoke.pop("matmul")
    assert all(smoke_matmul[axis] < matmul[axis] for axis in "mnk")
    assert smoke == pinned
    assert smoke_work.accounted_bytes * 100 < pinned_work.accounted_bytes
    assert smoke_work.peak_allocation_bytes * 50 < pinned_work.peak_allocation_bytes


def test_ci_gates_the_lab_and_operates_the_installed_command():
    job = ci_job("memory-hierarchy-cpu")

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
        f'ai-journey-memory-hierarchy run --config "$GITHUB_WORKSPACE/{SMOKE}"',
        f'--config "$GITHUB_WORKSPACE/{SMOKE}"',
        "name: memory-hierarchy-evidence",
    ):
        assert command in job, command
    operations = [
        shlex.split(line)
        for line in job.splitlines()
        if line.strip().startswith("ai-journey-memory-hierarchy ")
    ]
    assert [words[1] for words in operations] == ["run", "verify"]
    assert all(
        words[words.index("--config") + 1].endswith(SMOKE) for words in operations
    )
    run, verify = operations
    assert verify[verify.index("--report") + 1] == run[run.index("--output") + 1]
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern
