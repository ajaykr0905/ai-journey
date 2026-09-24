from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_day_21.py"


class Day21CliTests(unittest.TestCase):
    def test_cli_writes_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            json_output = Path(directory) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--smoothing",
                    "0.5",
                    "--output",
                    str(output),
                    "--json-output",
                    str(json_output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated Day 21", result.stderr)
            self.assertIn("Boundary audit", output.read_text(encoding="utf-8"))
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["smoothing"], 0.5)
            self.assertEqual(len(payload["evaluations"]), 4)
            self.assertEqual(payload["metrics"]["shifted_records"], 30)

    def test_cli_reports_invalid_smoothing_without_traceback(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--smoothing", "0"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Day 21 validation failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_stdout_mode_keeps_summary_on_stderr(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("# Day 21"))
        self.assertIn("Validated Day 21", result.stderr)


if __name__ == "__main__":
    unittest.main()
