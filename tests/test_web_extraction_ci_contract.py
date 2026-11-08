from pathlib import Path


def test_ci_gates_html_extraction_source_format_and_tests():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Check replayable HTML extraction contracts" in workflow
    assert "ruff check src/ai_journey/web_*.py" in workflow
    assert "ruff format --check src/ai_journey/web_*.py" in workflow
    assert (
        "python -m pytest -q tests/test_web_*.py tests/test_html_extraction.py"
        in workflow
    )
