from pathlib import Path
import os
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_training_command_is_declared_as_an_installed_entry_point():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["ai-journey-gpt2-train"] == (
        "ai_journey.gpt2_training_cli:main"
    )


def test_training_module_starts_in_a_fresh_python_process():
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "ai_journey.gpt2_training_cli", "--help"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "deterministic GPT-2 training" in result.stdout
    assert "{train,verify,replay}" in result.stdout
