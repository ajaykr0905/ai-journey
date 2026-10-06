from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day32RunnerTests(unittest.TestCase):
    def test_runner_writes_verified_hierarchical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-32.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_32.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--steps",
                    "4",
                    "--batch-size",
                    "8",
                    "--context-size",
                    "4",
                    "--embedding-dim",
                    "4",
                    "--hidden-dim",
                    "12",
                    "--group-factors",
                    "2",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["completed_steps"], 4)
        self.assertEqual(
            [step["output_shape"] for step in payload["shapes"]],
            [
                [2, 4, 4],
                [2, 2, 12],
                [2, 1, 12],
                [2, payload["model_config"]["vocab_size"]],
            ],
        )
        self.assertIn("report_fingerprint", payload)
        self.assertIn("train_nll=", completed.stdout)

    def test_invalid_hierarchy_preserves_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-32.json"
            output.write_bytes(b"previous evidence\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_32.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--context-size",
                    "4",
                    "--group-factors",
                    "2",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            final_bytes = output.read_bytes()

        self.assertEqual(completed.returncode, 2)
        self.assertIn("product of group_factors", completed.stderr)
        self.assertEqual(final_bytes, b"previous evidence\n")


if __name__ == "__main__":
    unittest.main()
