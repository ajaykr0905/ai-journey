from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Day24RunnerTests(unittest.TestCase):
    def test_invalid_numeric_controls_cannot_replace_existing_artifacts(self) -> None:
        for option in (
            "initialization-std",
            "learning-rate",
            "gradient-clip",
            "weight-decay",
        ):
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "report.json"
                checkpoint = root / "checkpoint.pt"
                output.write_bytes(b"previous report")
                checkpoint.write_bytes(b"previous checkpoint")
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_day_24.py"),
                        "--corpus",
                        str(ROOT / "data" / "day-19-demo-names.txt"),
                        "--output",
                        str(output),
                        "--checkpoint",
                        str(checkpoint),
                        f"--{option}=nan",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(f"{option.replace('-', '_')} must be", completed.stderr)
                self.assertIn("finite", completed.stderr)
                self.assertEqual(output.read_bytes(), b"previous report")
                self.assertEqual(checkpoint.read_bytes(), b"previous checkpoint")

    def test_runner_creates_checkpoint_and_json_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_day_24.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
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
                    "--initialization-std",
                    "0.03",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(payload["completed_steps"], 2)
            self.assertEqual(payload["model_config"]["initialization_std"], 0.03)
            self.assertIn("train_nll=", completed.stdout)

    def test_overfit_probe_uses_the_requested_example_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overfit.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_day_24_overfit.py"),
                    "--corpus",
                    str(ROOT / "data" / "day-19-demo-names.txt"),
                    "--output",
                    str(output),
                    "--examples",
                    "10",
                    "--steps",
                    "20",
                    "--batch-size",
                    "5",
                    "--embedding-dim",
                    "8",
                    "--heads",
                    "2",
                    "--layers",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["example_count"], 10)
        self.assertLess(payload["final_nll"], payload["initial_nll"])


if __name__ == "__main__":
    unittest.main()
