import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_runs_static_checks_comparison_and_exact_replay():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    for command in (
        "ruff check src/ai_journey/preference*.py tests/test_preference*.py",
        "python -m pytest -q tests/test_preference*.py",
        "ai-journey-preference-compare --config config/preference-comparison.json",
        "ai-journey-preference-verify --report",
    ):
        assert command in workflow


def test_readiness_requires_the_complete_preference_surface():
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]

    for path in (
        "src/ai_journey/preference_data.py",
        "src/ai_journey/preference_optimization.py",
        "src/ai_journey/preference_experiment.py",
        "src/ai_journey/preference_report.py",
        "src/ai_journey/preference_cli.py",
        "config/preference-comparison.json",
        "docs/preference-optimization.md",
        "tests/test_preference_data.py",
        "tests/test_preference_optimization.py",
        "tests/test_preference_experiment.py",
        "tests/test_preference_report.py",
        "tests/test_preference_cli.py",
        "tests/test_preference_package.py",
        "tests/test_preference_ci_contract.py",
    ):
        assert path in required
        assert (ROOT / path).is_file()
