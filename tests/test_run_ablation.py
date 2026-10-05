from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AblationRunnerCliTests(unittest.TestCase):
    def command(self, output: Path, protocol: Path | None = None) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "scripts" / "run_ablation.py"),
            "--protocol",
            str(protocol or ROOT / "config" / "day-31-learning-rate-ablation.json"),
            "--corpus",
            str(ROOT / "data" / "day-19-demo-names.txt"),
            "--output",
            str(output),
        ]

    def test_cli_runs_three_predeclared_arms_reproducibly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            completed = subprocess.run(
                self.command(first), check=True, capture_output=True, text=True
            )
            subprocess.run(
                self.command(second), check=True, capture_output=True, text=True
            )
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            payload = json.loads(first_bytes)

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(len(payload["result"]["protocol"]["arms"]), 3)
        self.assertEqual(len(payload["result"]["trials"]), 6)
        self.assertEqual(payload["result"]["protocol"]["trial_seeds"], [31, 32])
        self.assertTrue(payload["evaluation"]["complete_pairs"])
        self.assertIn("paired_seeds=2", completed.stdout)

    def test_invalid_protocol_preserves_an_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "invalid.json"
            payload = json.loads(
                (ROOT / "config" / "day-31-learning-rate-ablation.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["training_config"]["unexpected"] = 1
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "report.json"
            output.write_bytes(b"previous report")
            completed = subprocess.run(
                self.command(output, protocol),
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("fields do not match schema", completed.stderr)
            self.assertEqual(output.read_bytes(), b"previous report")


if __name__ == "__main__":
    unittest.main()
