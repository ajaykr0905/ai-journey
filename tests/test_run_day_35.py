from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_journey import day35_cli as run_day_35
from ai_journey.causal_experiment import verify_experiment_report


class Day35CLITests(unittest.TestCase):
    def test_cli_publishes_verified_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory, "nested", "day-35.json")
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = run_day_35.main(
                    [
                        "--output",
                        str(output),
                        "--batch-size",
                        "2",
                        "--time",
                        "5",
                        "--channels",
                        "3",
                        "--seed",
                        "3505",
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            verify_experiment_report(report)
            self.assertEqual(report["config"]["seed"], 3505)
            self.assertIn("max_forward_error=", stdout.getvalue())

    def test_invalid_controls_preserve_existing_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory, "day-35.json")
            output.write_text("preserve", encoding="utf-8")
            with redirect_stderr(StringIO()):
                code = run_day_35.main(["--output", str(output), "--time", "0"])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
