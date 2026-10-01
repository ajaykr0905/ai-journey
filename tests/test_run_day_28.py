"""The evidence command must fail closed and publish a complete JSON report."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Day28CommandTests(unittest.TestCase):
    def run_cli(self, path, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_day_28.py"),
                "--output",
                str(path),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_writes_repeatable_evidence_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "nested" / "first.json"
            second = Path(directory) / "second.json"
            result = self.run_cli(first)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.run_cli(second).returncode, 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            report = json.loads(first.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["environment"]["device"], "cpu")
            self.assertEqual(len(report["cases"]), 3)
            self.assertGreater(len(report["transformer"]["parameter_errors"]), 10)
            self.assertEqual(list(first.parent.iterdir()), [first])

    def test_failed_gate_still_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            result = self.run_cli(path, "--tolerance", "0")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse(json.loads(path.read_text())["passed"])

    def test_invalid_controls_do_not_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("original")
            for extra in [
                ("--epsilon", "0"),
                ("--tolerance", "nan"),
                ("--seed", "-1"),
                ("--transformer-tolerance", "-1"),
            ]:
                with self.subTest(extra=extra):
                    result = self.run_cli(path, *extra)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
