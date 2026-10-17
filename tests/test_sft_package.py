import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from ai_journey.sft_cli import load_config
from ai_journey.sft_experiment import SFTComparisonConfig

ROOT = Path(__file__).resolve().parents[1]


class SFTPackageContractTests(unittest.TestCase):
    def test_package_exports_compare_and_verify_commands(self):
        configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
        scripts = configuration["project"]["scripts"]
        self.assertEqual(
            scripts["ai-journey-chat-sft-compare"], "ai_journey.sft_cli:main"
        )
        self.assertEqual(
            scripts["ai-journey-chat-sft-verify"], "ai_journey.sft_cli:verify_main"
        )

    def test_checked_in_protocol_repeats_and_replays_in_fresh_processes(self):
        config = ROOT / "config/chat-sft-comparison.json"
        self.assertEqual(load_config(config), SFTComparisonConfig())
        config_before = config.read_bytes()
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/report.json"
            command = [
                sys.executable,
                "-m",
                "ai_journey.sft_cli",
                "--config",
                str(config),
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
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            before = output.read_bytes()
            second = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, output.read_bytes())
            verification = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from ai_journey.sft_cli import verify_main; raise SystemExit(verify_main())",
                    "--report",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(verification.returncode, 0, verification.stderr)
            self.assertIn("2 bases and 8 CPU chat SFT trials", verification.stdout)
            self.assertEqual(before, output.read_bytes())
            self.assertEqual(config_before, config.read_bytes())
            report = json.loads(before)
            self.assertEqual(report["result"]["heldout"]["target_tokens"], 593)
            for seed in report["result"]["seeds"]:
                self.assertEqual(seed["sft_schedule"]["input_tokens"], 1024)
                self.assertEqual(len(seed["trials"]), 4)
                for trial in seed["trials"]:
                    self.assertEqual(trial["heldout"]["target_tokens"], 593)


if __name__ == "__main__":
    unittest.main()
