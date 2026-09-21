from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_day_17 import main, parse_args


class Day17CliTests(unittest.TestCase):
    def test_parse_args_exposes_reproducibility_controls(self) -> None:
        args = parse_args(
            [
                "--steps",
                "25",
                "--learning-rate",
                "0.1",
                "--seed",
                "42",
                "--record-every",
                "5",
            ]
        )
        self.assertEqual(args.steps, 25)
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.record_every, 5)

    def test_cli_writes_validated_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            json_output = Path(directory) / "report.json"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main(
                    ["--output", str(output), "--json-output", str(json_output)]
                )
            self.assertEqual(status, 0)
            self.assertIn("Day 17 scalar MLP training report", output.read_text())
            payload = json.loads(json_output.read_text())
            self.assertEqual(payload["metrics"]["parameter_count"], 41)
            self.assertEqual(payload["metrics"]["final_accuracy"], 1.0)
            self.assertIn("Validated Day 17 scalar MLP", stderr.getvalue())

    def test_cli_prints_report_when_output_is_omitted(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = main([])
        self.assertEqual(status, 0)
        self.assertTrue(stdout.getvalue().startswith("# Day 17 scalar MLP"))

    def test_cli_returns_failure_for_invalid_training_settings(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["--steps", "0"])
        self.assertEqual(status, 1)
        self.assertIn("validation failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
