from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_day_19.py"


class Day19CliTests(unittest.TestCase):
    def test_cli_writes_markdown_full_json_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            json_output = Path(directory) / "report.json"
            manifest_output = Path(directory) / "manifest.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--samples",
                    "4",
                    "--seed",
                    "19",
                    "--output",
                    str(output),
                    "--json-output",
                    str(json_output),
                    "--manifest-output",
                    str(manifest_output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated Day 19", result.stderr)
            self.assertIn("terminated=4/4", result.stderr)
            self.assertIn("Data manifest", output.read_text(encoding="utf-8"))
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["sample_count"], 4)
            self.assertEqual(payload["metrics"]["seed"], 19)
            self.assertEqual(payload["manifest"], manifest)

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
            self.assertIn("Day 19 validation failed", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_stdout_mode_keeps_report_and_summary_on_separate_streams(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--samples", "2"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("# Day 19"))
        self.assertIn("Validated Day 19", result.stderr)


if __name__ == "__main__":
    unittest.main()
