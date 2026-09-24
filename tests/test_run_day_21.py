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
    def test_cli_writes_compact_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--output", str(output)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["baseline_records"], 20)
            self.assertEqual(payload["shifted_records"], 30)
            self.assertEqual(payload["shifted_boundary_leaks"], 29)
            self.assertNotIn("baseline_corpus", payload)

    def test_cli_reports_invalid_smoothing_without_traceback(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--smoothing", "0"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("corpus shift failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
