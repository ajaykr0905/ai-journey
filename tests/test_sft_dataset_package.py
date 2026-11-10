import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_registers_the_sft_dataset_operator_command():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        project["project"]["scripts"]["ai-journey-sft-dataset"]
        == "ai_journey.sft_dataset_cli:main"
    )


def test_packaged_module_exposes_build_verify_and_replay_help():
    result = subprocess.run(
        [sys.executable, "-m", "ai_journey.sft_dataset_cli", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "{build,verify,replay}" in result.stdout
    assert "reproducible supervised fine-tuning datasets" in result.stdout
