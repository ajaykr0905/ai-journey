from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_focused_scaling_tests_and_static_checks():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "scaling-plan-cpu:" in workflow
    assert "ruff check src/ai_journey/scaling_*.py tests/test_scaling_*.py" in workflow
    assert "python -m pytest -q tests/test_scaling_*.py" in workflow


def test_ci_builds_and_verifies_the_installed_pinned_plan():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "python -m build --wheel" in workflow
    assert "ai-journey-scaling build" in workflow
    assert "config/day-60-scaling-plan.json" in workflow
    assert "ai-journey-scaling verify" in workflow
    assert "scaling-plan-evidence" in workflow
