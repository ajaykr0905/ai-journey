import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_recomputes_and_verifies_the_pinned_roofline_report():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "ruff check src/ai_journey/roofline*.py tests/test_roofline*.py" in workflow
    assert "python -m pytest -q tests/test_roofline*.py" in workflow
    assert (
        "ai-journey-roofline build --config config/day-55-roofline-plan.json"
        in workflow
    )
    assert "ai-journey-roofline verify --report" in workflow


def test_readiness_requires_the_complete_roofline_surface():
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]
    paths = (
        "config/day-55-roofline-plan.json",
        "docs/roofline-analysis.md",
        "src/ai_journey/roofline.py",
        "src/ai_journey/roofline_cli.py",
        "src/ai_journey/roofline_report.py",
        "src/ai_journey/roofline_report_io.py",
        "src/ai_journey/roofline_transformer.py",
        "tests/test_roofline.py",
        "tests/test_roofline_cli.py",
        "tests/test_roofline_operations.py",
        "tests/test_roofline_report.py",
        "tests/test_roofline_report_io.py",
        "tests/test_roofline_transformer.py",
        "tests/test_roofline_ci_contract.py",
    )
    for path in paths:
        assert path in required
        assert (ROOT / path).is_file()
