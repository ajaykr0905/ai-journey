from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_ci_executes_and_verifies_installed_project1_operator():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    for command in (
        "ruff check src/ai_journey/project1_*.py tests/test_project1_*.py",
        "python -m pytest -q tests/test_project1_*.py",
        "ai-journey-project1 init --source",
        "ai-journey-project1 run --source",
        "ai-journey-project1 status --run-dir",
        "ai-journey-project1 verify --run-dir",
    ):
        assert command in workflow
    assert "name: project1-run-evidence" in workflow
