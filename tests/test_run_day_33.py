from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_journey.wavenet_rebuild import verify_rebuild_report

ROOT = Path(__file__).resolve().parents[1]


class Day33RunnerTests(unittest.TestCase):
    def test_runner_writes_verified_rebuild_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-33.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_33.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--steps",
                    "2",
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
                    "--overfit-steps",
                    "30",
                    "--minimum-overfit-improvement",
                    "0.3",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        verify_rebuild_report(payload)
        self.assertEqual(payload["completed_steps"], 2)
        self.assertTrue(payload["forward_audit"]["passed"])
        self.assertTrue(payload["gradient_audit"]["passed"])
        self.assertTrue(payload["finite_difference_audit"]["passed"])
        self.assertTrue(payload["overfit_probe"]["passed"])
        self.assertIn("forward_error=", completed.stdout)

    def test_invalid_hierarchy_preserves_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-33.json"
            output.write_bytes(b"previous evidence\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_33.py"),
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
