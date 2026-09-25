from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrainContextMLPTests(unittest.TestCase):
    def test_cli_trains_and_writes_reproducibility_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_context_mlp.py",
                    "--corpus",
                    "data/day-19-demo-names.txt",
                    "--output",
                    str(output),
                    "--steps",
                    "10",
                    "--seed",
                    "22",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertLess(payload["final_loss"], payload["initial_loss"])
            self.assertEqual(payload["steps"], 10)
            self.assertEqual(len(payload["model_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
