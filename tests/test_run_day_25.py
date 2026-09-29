from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day25RunnerTests(unittest.TestCase):
    def test_runner_writes_reproducible_report_and_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-25.json"
            plot = Path(directory) / "day-25.svg"
            gradient_plot = Path(directory) / "day-25-gradients.svg"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_day_25.py"),
                "--corpus",
                str(ROOT / "data" / "day-19-demo-names.txt"),
                "--output",
                str(output),
                "--plot",
                str(plot),
                "--gradient-plot",
                str(gradient_plot),
                "--steps",
                "1",
                "--batch-size",
                "4",
                "--block-size",
                "4",
                "--embedding-dim",
                "8",
                "--heads",
                "2",
                "--layers",
                "1",
                "--stressed-initialization-std",
                "0.8",
            ]
            environment = {
                **os.environ,
                "MPLCONFIGDIR": str(Path(directory) / "matplotlib"),
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            }
            first = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            first_report = output.read_bytes()
            first_plot = plot.read_bytes()
            first_gradient_plot = gradient_plot.read_bytes()
            second = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            second_report = output.read_bytes()
            second_plot = plot.read_bytes()
            second_gradient_plot = gradient_plot.read_bytes()
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_plot, second_plot)
        self.assertEqual(first_gradient_plot, second_gradient_plot)
        self.assertEqual(
            [item["name"] for item in payload["variants"]], ["baseline", "stressed"]
        )
        self.assertIn("baseline: initial_loss=", first.stdout)
        self.assertIn("stressed: initial_loss=", first.stdout)


if __name__ == "__main__":
    unittest.main()
