from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_journey.wavenet_certification import load_certification_report

ROOT = Path(__file__).resolve().parents[1]


class Day33CertificationRunnerTests(unittest.TestCase):
    def command(self, output: Path) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "scripts" / "certify_day_33.py"),
            "--corpus",
            str(ROOT / "data" / "day-19-demo-names.txt"),
            "--output",
            str(output),
            "--examples",
            "4",
        ]

    def test_runner_writes_verified_certification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "certification.json"
            completed = subprocess.run(
                self.command(output), check=True, capture_output=True, text=True
            )
            payload = load_certification_report(output)

        self.assertEqual(payload["curriculum_day"], 33)
        self.assertTrue(all(payload["gates"].values()))
        self.assertIn("gates=13", completed.stdout)

    def test_invalid_examples_preserve_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "certification.json"
            output.write_bytes(b"previous evidence\n")
            command = self.command(output)
            command[-1] = "0"
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("examples must be positive", completed.stderr)
            self.assertEqual(output.read_bytes(), b"previous evidence\n")


if __name__ == "__main__":
    unittest.main()
