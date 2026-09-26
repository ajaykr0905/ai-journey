from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day23CLITests(unittest.TestCase):
    def test_cli_writes_a_reproducible_sweep_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-23.json"
            checkpoint = Path(directory) / "day-23-model.json"
            command = [
                sys.executable,
                "scripts/run_day_23.py",
                "--corpus",
                "data/day-19-demo-names.txt",
                "--output",
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--rate-count",
                "3",
                "--seed",
                "23",
            ]
            first = subprocess.run(
                command, cwd=ROOT, check=False, capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            first_payload = json.loads(output.read_text(encoding="utf-8"))
            second = subprocess.run(
                command, cwd=ROOT, check=False, capture_output=True, text=True
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            second_payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(first_payload, second_payload)
            self.assertEqual(len(first_payload["trials"]), 3)
            self.assertIn("test", first_payload["metrics"])
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["model_fingerprint"],
                first_payload["selected_model_fingerprint"],
            )
            selected = next(
                trial
                for trial in first_payload["trials"]
                if trial["learning_rate"]
                == first_payload["selected_learning_rate"]
            )
            self.assertEqual(saved["step"], selected["best_epoch"] + 1)


if __name__ == "__main__":
    unittest.main()
