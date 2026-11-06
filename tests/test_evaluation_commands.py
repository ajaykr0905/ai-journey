from pathlib import Path
import tomllib

import pytest

from ai_journey.evaluation_suite_cli import main as suite_main
from ai_journey.lambada_cli import main as lambada_main


def test_package_exposes_lambada_and_suite_commands():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    assert scripts["ai-journey-lambada"] == "ai_journey.lambada_cli:main"
    assert scripts["ai-journey-evaluation-suite"] == (
        "ai_journey.evaluation_suite_cli:main"
    )


@pytest.mark.parametrize("command", [lambada_main, suite_main])
def test_evaluation_commands_expose_help(command, capsys):
    with pytest.raises(SystemExit) as result:
        command(["--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out
