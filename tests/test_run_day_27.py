from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_day_27


class Day27RunnerTests(unittest.TestCase):
    def _arguments(self, root: Path) -> list[str]:
        corpus = root / "corpus.txt"
        corpus.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
        return [
            "--corpus",
            str(corpus),
            "--output",
            str(root / "nested" / "report.json"),
            "--plot",
            str(root / "nested" / "diagnostics.svg"),
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
            "--min-training-loss-reduction",
            "0",
            "--min-mode-nll-gap",
            "0",
            "--min-train-batch-coupling",
            "0",
        ]

    def test_runner_writes_reproducible_report_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            self.assertEqual(run_day_27.main(arguments), 0)
            report_path = root / "nested" / "report.json"
            plot_path = root / "nested" / "diagnostics.svg"
            first_report = report_path.read_bytes()
            first_plot = plot_path.read_bytes()
            self.assertEqual(run_day_27.main(arguments), 0)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(first_report, report_path.read_bytes())
            self.assertEqual(first_plot, plot_path.read_bytes())
        self.assertTrue(payload["evaluation"]["passed"])
        self.assertEqual(
            payload["experiment"]["model_config"]["normalization_mode"],
            "scratch_batch_norm",
        )
        self.assertEqual(len(payload["report_fingerprint"]), 64)

    def test_runner_returns_one_when_a_predeclared_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            arguments.extend(["--min-mode-nll-gap", "100"])
            self.assertEqual(run_day_27.main(arguments), 1)
            payload = json.loads(
                (root / "nested" / "report.json").read_text(encoding="utf-8")
            )
        self.assertIn("mode_nll_gap", payload["evaluation"]["violations"])


if __name__ == "__main__":
    unittest.main()
