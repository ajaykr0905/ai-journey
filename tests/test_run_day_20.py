from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_day_20.py"


class Day20CliTests(unittest.TestCase):
    def test_cli_writes_markdown_and_full_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            json_output = Path(directory) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--steps",
                    "40",
                    "--samples",
                    "4",
                    "--seed",
                    "20",
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
            self.assertIn("Validated Day 20", result.stderr)
            self.assertIn("count_neural_error=0.000e+00", result.stderr)
            self.assertIn("Count-to-neural equivalence", output.read_text())
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["sample_count"], 4)
            self.assertEqual(payload["metrics"]["seed"], 20)
            self.assertEqual(len(payload["gradient_probes"]), 6)

    def test_cli_reports_invalid_smoothing_without_traceback(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--smoothing", "0"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Day 20 validation failed", result.stderr)
        self.assertIn("smoothing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_reports_invalid_corpus_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "invalid.txt"
            corpus.write_text("valid\nNotLowercase\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--corpus", str(corpus)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Day 20 validation failed", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_stdout_mode_separates_report_and_summary_streams(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--steps", "10", "--samples", "2"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("# Day 20"))
        self.assertIn("Validated Day 20", result.stderr)


if __name__ == "__main__":
    unittest.main()
