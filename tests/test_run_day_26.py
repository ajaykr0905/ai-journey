from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day26RunnerTests(unittest.TestCase):
    def test_runner_writes_reproducible_report_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "day-26.json"
            loss_plot = Path(directory) / "day-26-loss.svg"
            audit_plot = Path(directory) / "day-26-audit.svg"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_day_26.py"),
                "--corpus",
                str(ROOT / "data" / "day-19-demo-names.txt"),
                "--output",
                str(output),
                "--loss-plot",
                str(loss_plot),
                "--audit-plot",
                str(audit_plot),
                "--steps",
                "10",
                "--batch-size",
                "8",
                "--block-size",
                "8",
                "--embedding-dim",
                "16",
                "--heads",
                "4",
                "--layers",
                "1",
                "--kaiming-gain",
                "1.0",
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
            first_files = (
                output.read_bytes(),
                loss_plot.read_bytes(),
                audit_plot.read_bytes(),
            )
            second = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            second_files = (
                output.read_bytes(),
                loss_plot.read_bytes(),
                audit_plot.read_bytes(),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(first_files, second_files)
        self.assertTrue(payload["evaluation"]["passed"])
        self.assertEqual(
            [item["name"] for item in payload["experiment"]["variants"]],
            ["fixed_normal", "kaiming_normal"],
        )
        self.assertIn("kaiming_mean_loss_improvement=", first.stdout)
        self.assertIn("kaiming_validation_improved=", first.stdout)
        self.assertIn("passed=True", first.stdout)

    def test_runner_returns_one_when_a_predeclared_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_26.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(Path(directory) / "day-26.json"),
                    "--loss-plot",
                    str(Path(directory) / "day-26-loss.svg"),
                    "--audit-plot",
                    str(Path(directory) / "day-26-audit.svg"),
                    "--steps",
                    "2",
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
                    "--max-relative-std-error",
                    "0",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "MPLCONFIGDIR": str(Path(directory) / "matplotlib"),
                    "XDG_CACHE_HOME": str(Path(directory) / "cache"),
                },
            )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("initialization_std_error", completed.stdout)
        self.assertIn("passed=False", completed.stdout)


if __name__ == "__main__":
    unittest.main()
