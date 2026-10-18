import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_preference_compare_and_verify_commands():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = configuration["project"]["scripts"]

    assert scripts["ai-journey-preference-compare"] == "ai_journey.preference_cli:main"
    assert (
        scripts["ai-journey-preference-verify"]
        == "ai_journey.preference_cli:verify_main"
    )
