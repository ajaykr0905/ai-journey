from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_focused_inference_tests_and_static_checks():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  inference-plan-cpu:\n", 1)[1]

    assert "python -m pip install torch --index-url" in job
    assert 'python -m pip install "numpy>=1.24,<2"' in job
    assert "ruff check src/ai_journey/inference_*.py tests/test_inference_*.py" in job
    assert "python -m pytest -q tests/test_inference_*.py" in job


def test_ci_builds_and_verifies_the_installed_pinned_inference_plan():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  inference-plan-cpu:\n", 1)[1]

    assert "python -m build --wheel" in job
    assert "ai-journey-inference build" in job
    assert "config/day-61-inference-plan.json" in job
    assert "ai-journey-inference verify" in job
    assert "inference-plan-evidence" in job
