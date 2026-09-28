from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day24RunnerTests(unittest.TestCase):
    def test_runner_creates_checkpoint_and_json_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_24.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                    "--steps",
                    "2",
                    "--batch-size",
                    "4",
                    "--block-size",
                    "4",
                    "--embedding-dim",
                    "8",
                    "--heads",
                    "2",
                    "--layers",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(payload["completed_steps"], 2)
            self.assertIn("train_nll=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
