from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_package_exposes_installed_recovery_command():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'ai-journey-recovery = "ai_journey.recovery_cli:main"' in pyproject
