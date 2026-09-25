from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_corpus_shift.py"
BASELINE = ROOT / "data" / "day-19-demo-names.txt"
CANDIDATE = ROOT / "data" / "day-21-indian-cities.txt"


class CorpusShiftCliTests(unittest.TestCase):
    def run_check(self, *policy: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline",
                str(BASELINE),
                "--candidate",
                str(CANDIDATE),
                *policy,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_passed_policy_returns_zero_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = self.run_check(
                "--max-js-divergence",
                "0.05",
                "--max-perplexity-ratio",
                "2.5",
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["violations"], [])

    def test_violated_policy_returns_two(self) -> None:
        result = self.run_check(
            "--max-js-divergence",
            "0.01",
            "--max-perplexity-ratio",
            "2.0",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["passed"])
        self.assertEqual(len(payload["violations"]), 2)

    def test_invalid_policy_returns_one_without_traceback(self) -> None:
        result = self.run_check(
            "--max-js-divergence",
            "0",
            "--max-perplexity-ratio",
            "2.0",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("corpus shift check failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
