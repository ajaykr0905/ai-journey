import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_recomputes_and_verifies_the_pinned_resource_plan():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "ruff check src/ai_journey/resource*.py tests/test_resource*.py" in workflow
    assert "python -m pytest -q tests/test_resource*.py" in workflow
    assert (
        "ai-journey-resource-plan --config config/day-51-resource-plan.json" in workflow
    )
    assert "ai-journey-resource-verify --report" in workflow


def test_readiness_requires_the_complete_resource_planning_surface():
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]
    paths = (
        "config/day-51-resource-plan.json",
        "docs/resource-accounting.md",
        "src/ai_journey/resource_accounting.py",
        "src/ai_journey/resource_cli.py",
        "src/ai_journey/resource_report.py",
        "src/ai_journey/resource_report_io.py",
        "tests/test_resource_accounting.py",
        "tests/test_resource_cli.py",
        "tests/test_resource_report.py",
        "tests/test_resource_report_io.py",
        "tests/test_resource_ci_contract.py",
    )
    for path in paths:
        assert path in required
        assert (ROOT / path).is_file()
