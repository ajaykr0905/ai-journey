from pathlib import Path
import tomllib


def test_package_installs_html_extraction_command():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["ai-journey-html-extract"] == (
        "ai_journey.web_extraction_cli:main"
    )
