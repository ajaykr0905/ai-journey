from __future__ import annotations

import unittest

from ai_journey.causal_average import CausalAverageError
from ai_journey.causal_experiment import CausalExperimentConfig, run_causal_experiment


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


if __name__ == "__main__":
    unittest.main()
