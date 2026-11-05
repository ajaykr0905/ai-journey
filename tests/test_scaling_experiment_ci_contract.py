from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_checks_the_scaling_experiment_surface():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    scaling_job = workflow.split("  scaling-plan-cpu:\n", 1)[1].split(
        "  inference-plan-cpu:\n", 1
    )[0]

    assert (
        "ruff check src/ai_journey/scaling_*.py tests/test_scaling_*.py" in scaling_job
    )
    assert "python -m pytest -q tests/test_scaling_*.py" in scaling_job


def test_ci_builds_and_verifies_installed_experiment_evidence():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    scaling_job = workflow.split("  scaling-plan-cpu:\n", 1)[1].split(
        "  inference-plan-cpu:\n", 1
    )[0]

    assert "ai-journey-scaling-experiment build" in scaling_job
    assert "config/day-62-scaling-experiment-fixture.json" in scaling_job
    assert "experiment-report.json" in scaling_job
    assert "ai-journey-scaling-experiment verify" in scaling_job
