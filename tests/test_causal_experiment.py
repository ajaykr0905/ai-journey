from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_journey.causal_average import CausalAverageError
from ai_journey.causal_experiment import (
    CausalExperimentConfig,
    build_experiment_report,
    run_causal_experiment,
    verify_experiment_report,
    write_experiment_report,
)


class CausalExperimentTests(unittest.TestCase):
    def test_replays_the_same_complete_evidence(self) -> None:
        config = CausalExperimentConfig(batch_size=4, time=9, channels=5, seed=3509)
        first = run_causal_experiment(config)
        second = run_causal_experiment(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first.input_sha256), 64)

    def test_every_measured_error_passes_the_declared_tolerance(self) -> None:
        result = run_causal_experiment()
        tolerance = result.config.tolerance
        errors = (
            result.forward.matmul_error,
            result.forward.softmax_error,
            result.forward.cumsum_error,
            result.gradient.matmul_error,
            result.gradient.softmax_error,
            result.gradient.cumsum_error,
            result.loop_future_error,
            result.matmul_future_error,
            result.softmax_future_error,
            result.cumsum_future_error,
            result.padding_error,
            result.streaming_error,
        )
        self.assertTrue(all(error <= tolerance for error in errors))

    def test_rejects_unbounded_or_malformed_controls(self) -> None:
        with self.assertRaisesRegex(CausalAverageError, "time must be positive"):
            CausalExperimentConfig(time=0)
        with self.assertRaisesRegex(TypeError, "seed must be an integer"):
            CausalExperimentConfig(seed=True)
        with self.assertRaisesRegex(CausalAverageError, "finite and non-negative"):
            CausalExperimentConfig(tolerance=float("nan"))

    def test_supports_the_minimum_one_position_experiment(self) -> None:
        result = run_causal_experiment(
            CausalExperimentConfig(batch_size=1, time=1, channels=1)
        )
        self.assertEqual(result.streaming_error, 0.0)


class CausalExperimentReportTests(unittest.TestCase):
    def test_report_round_trips_through_atomic_json(self) -> None:
        report = build_experiment_report(run_causal_experiment())
        verify_experiment_report(report)
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "day-35.json")
            write_experiment_report(path, report)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, report)
            verify_experiment_report(loaded)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_report_detects_metric_tampering(self) -> None:
        report = build_experiment_report(run_causal_experiment())
        report["streaming_error"] = 1.0
        with self.assertRaisesRegex(CausalAverageError, "fingerprint mismatch"):
            verify_experiment_report(report)

    def test_failed_publication_preserves_existing_report(self) -> None:
        report = build_experiment_report(run_causal_experiment())
        with TemporaryDirectory() as directory:
            path = Path(directory, "day-35.json")
            path.write_text("preserve", encoding="utf-8")
            with patch(
                "ai_journey.causal_experiment.os.replace", side_effect=OSError("boom")
            ):
                with self.assertRaisesRegex(OSError, "boom"):
                    write_experiment_report(path, report)
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
