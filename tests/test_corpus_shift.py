from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.corpus_shift import (
    CorpusShiftError,
    ShiftPolicy,
    analyze_corpus_shift,
    assess_corpus_shift,
    assessment_payload,
    boundary_leaks,
    build_shared_vocabulary,
    evaluate,
    jensen_shannon,
    report_payload,
    smoothed_probabilities,
)

BASELINE = ("anna", "aria", "navi")
SHIFTED = ("agra", "mumbai", "pune")


class CorpusShiftTests(unittest.TestCase):
    def test_shared_vocabulary_is_boundary_first_and_sorted(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        expected = tuple(sorted(set("".join(BASELINE + SHIFTED))))
        self.assertEqual(vocabulary.tokens, (".", *expected))

    def test_smoothed_probabilities_are_positive_unit_rows(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        probabilities = smoothed_probabilities(BASELINE, vocabulary, smoothing=0.5)
        self.assertTrue(np.all(probabilities > 0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluation_matches_hand_computed_loss(self) -> None:
        vocabulary = build_shared_vocabulary(("aa",), ("aa",))
        probabilities = smoothed_probabilities(("aa",), vocabulary)
        result = evaluate(
            ("aa",),
            vocabulary,
            probabilities,
            trained_on="toy",
            evaluated_on="toy",
        )
        expected = -(math.log(2 / 3) + math.log(1 / 2) + math.log(1 / 2)) / 3
        self.assertAlmostEqual(result.mean_nll, expected)
        self.assertAlmostEqual(result.perplexity, math.exp(expected))
        self.assertEqual(result.transitions, 3)

    def test_evaluation_rejects_invalid_probability_matrix(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        with self.assertRaises(CorpusShiftError):
            evaluate(
                BASELINE,
                vocabulary,
                np.ones((2, 2), dtype=np.float64),
                trained_on="baseline",
                evaluated_on="baseline",
            )

    def test_jensen_shannon_is_symmetric_and_zero_for_identity(self) -> None:
        left = np.array([0.8, 0.2], dtype=np.float64)
        right = np.array([0.1, 0.9], dtype=np.float64)
        self.assertEqual(jensen_shannon(left, left), 0.0)
        self.assertAlmostEqual(jensen_shannon(left, right), jensen_shannon(right, left))

    def test_jensen_shannon_rejects_invalid_distributions(self) -> None:
        with self.assertRaises(CorpusShiftError):
            jensen_shannon(
                np.array([1.0, -1.0], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            )

    def test_boundary_leaks_are_exact(self) -> None:
        self.assertEqual(
            boundary_leaks(("agra", "mumbai", "pune")),
            (("a", "m"), ("i", "p")),
        )

    def test_analysis_returns_complete_cross_evaluation(self) -> None:
        report = analyze_corpus_shift(BASELINE, SHIFTED)
        pairs = {
            (result.trained_on, result.evaluated_on) for result in report.evaluations
        }
        self.assertEqual(
            pairs,
            {
                ("baseline", "baseline"),
                ("baseline", "shifted"),
                ("shifted", "baseline"),
                ("shifted", "shifted"),
            },
        )

    def test_analysis_is_deterministic(self) -> None:
        first = analyze_corpus_shift(BASELINE, SHIFTED)
        second = analyze_corpus_shift(BASELINE, SHIFTED)
        self.assertEqual(first, second)

    def test_analysis_rejects_invalid_smoothing(self) -> None:
        for value in (True, 0, -1, float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, CorpusShiftError)),
            ):
                analyze_corpus_shift(BASELINE, SHIFTED, smoothing=value)

    def test_policy_reports_both_drift_violations(self) -> None:
        report = analyze_corpus_shift(BASELINE, SHIFTED)
        assessment = assess_corpus_shift(
            report,
            ShiftPolicy(max_js_divergence=1e-9, max_perplexity_ratio=1e-9),
        )
        self.assertFalse(assessment.passed)
        self.assertEqual(
            assessment.violations,
            (
                "transition_js_divergence",
                "baseline_to_shifted_perplexity_ratio",
            ),
        )

    def test_policy_passes_when_metrics_are_within_limits(self) -> None:
        report = analyze_corpus_shift(BASELINE, SHIFTED)
        assessment = assess_corpus_shift(
            report,
            ShiftPolicy(max_js_divergence=1.0, max_perplexity_ratio=10.0),
        )
        payload = assessment_payload(report, assessment)
        self.assertTrue(assessment.passed)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["violations"], ())

    def test_policy_rejects_invalid_limits(self) -> None:
        for value in (True, 0, -1, float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, CorpusShiftError)),
            ):
                ShiftPolicy(
                    max_js_divergence=value,
                    max_perplexity_ratio=1.0,
                )

    def test_payload_is_compact(self) -> None:
        payload = report_payload(analyze_corpus_shift(BASELINE, SHIFTED))
        self.assertEqual(payload["baseline_records"], 3)
        self.assertEqual(payload["shifted_records"], 3)
        self.assertEqual(payload["baseline_boundary_leaks"], 2)
        self.assertEqual(payload["shifted_boundary_leaks"], 2)
        self.assertNotIn("baseline_corpus", payload)
        self.assertNotIn("probabilities", payload)


if __name__ == "__main__":
    unittest.main()
