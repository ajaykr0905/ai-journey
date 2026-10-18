import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from ai_journey.preference_experiment import (
    PreferenceComparisonConfig,
    preference_config_dict,
)


ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_preference_compare_and_verify_commands():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = configuration["project"]["scripts"]

    assert scripts["ai-journey-preference-compare"] == "ai_journey.preference_cli:main"
    assert (
        scripts["ai-journey-preference-verify"]
        == "ai_journey.preference_cli:verify_main"
    )


def test_preference_protocol_repeats_and_replays_in_fresh_processes():
    config = PreferenceComparisonConfig(fold_count=2, sft_steps=1, preference_steps=1)
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.json"
        output = root / "nested/report.json"
        config_path.write_text(json.dumps(preference_config_dict(config)))
        config_before = config_path.read_bytes()
        command = [
            sys.executable,
            "-m",
            "ai_journey.preference_cli",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ]
        first = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert first.returncode == 0, first.stderr
        before = output.read_bytes()
        second = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert second.returncode == 0, second.stderr
        assert output.read_bytes() == before
        verification = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from ai_journey.preference_cli import verify_main; "
                    "raise SystemExit(verify_main())"
                ),
                "--report",
                str(output),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert verification.returncode == 0, verification.stderr
        assert "16 trials" in verification.stdout
        assert output.read_bytes() == before
        assert config_path.read_bytes() == config_before
