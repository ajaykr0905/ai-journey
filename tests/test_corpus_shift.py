from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.corpus_shift import (
    CorpusShiftValidationError,
    audit_boundaries,
    build_shared_vocabulary,
    character_jaccard_similarity,
    character_set,
    cross_record_transitions,
    evaluate_corpus,
    experiment_metrics,
    experiment_payload,
    jensen_shannon_divergence,
    packed_stream_bigrams,
    render_boundary_mermaid,
    render_corpus_shift_markdown,
    run_corpus_shift_experiment,
    smoothed_bigram_probabilities,
    smoothed_joint_distribution,
    unseen_characters,
    validate_experiment,
    validate_probability_matrix,
    validate_smoothing,
)

BASELINE = ("anna", "aria", "navi")
SHIFTED = ("agra", "mumbai", "pune")


class CorpusShiftTests(unittest.TestCase):
    def test_validate_smoothing_accepts_positive_real(self) -> None:
        self.assertEqual(validate_smoothing(0.5), 0.5)

    def test_validate_smoothing_rejects_bool_zero_and_non_finite(self) -> None:
        for value in (True, 0.0, -1.0, float("inf"), float("nan")):
            with (
                self.subTest(value=value),
                self.assertRaises((TypeError, CorpusShiftValidationError)),
            ):
                validate_smoothing(value)

    def test_character_set_is_normalized_and_excludes_boundary(self) -> None:
        self.assertEqual(character_set(("anna", "aria")), frozenset("anri"))

    def test_unseen_characters_are_sorted(self) -> None:
        self.assertEqual(
            unseen_characters(("anna",), ("mumbai",)), ("b", "i", "m", "u")
        )

    def test_character_jaccard_similarity_matches_set_definition(self) -> None:
        expected = len(character_set(BASELINE) & character_set(SHIFTED)) / len(
            character_set(BASELINE) | character_set(SHIFTED)
        )
        self.assertEqual(character_jaccard_similarity(BASELINE, SHIFTED), expected)

    def test_shared_vocabulary_is_boundary_first_and_sorted(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        self.assertEqual(vocabulary.tokens[0], ".")
        self.assertEqual(
            vocabulary.tokens[1:], tuple(sorted(set("".join(BASELINE + SHIFTED))))
        )

    def test_smoothed_probabilities_are_positive_unit_rows(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        probabilities = smoothed_bigram_probabilities(
            BASELINE, vocabulary, smoothing=0.25
        )
        self.assertEqual(probabilities.shape, (vocabulary.size, vocabulary.size))
        self.assertTrue(np.all(probabilities > 0.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    def test_probability_validation_rejects_bad_shape_and_rows(self) -> None:
        with self.assertRaises(CorpusShiftValidationError):
            validate_probability_matrix(np.ones((2, 3), dtype=np.float64), 2)
        with self.assertRaises(CorpusShiftValidationError):
            validate_probability_matrix(np.zeros((2, 2), dtype=np.float64), 2)

    def test_evaluation_matches_hand_computed_toy_loss(self) -> None:
        vocabulary = build_shared_vocabulary(("aa",), ("aa",))
        probabilities = smoothed_bigram_probabilities(("aa",), vocabulary)
        evaluation = evaluate_corpus(
            ("aa",),
            vocabulary,
            probabilities,
            trained_on="toy",
            evaluated_on="toy",
        )
        expected = -(math.log(2.0 / 3.0) + math.log(0.5) + math.log(0.5)) / 3.0
        self.assertAlmostEqual(evaluation.mean_nll, expected)
        self.assertAlmostEqual(evaluation.perplexity, math.exp(expected))
        self.assertEqual(evaluation.transition_count, 3)

    def test_joint_distribution_is_flat_and_normalized(self) -> None:
        vocabulary = build_shared_vocabulary(BASELINE, SHIFTED)
        distribution = smoothed_joint_distribution(BASELINE, vocabulary)
        self.assertEqual(distribution.shape, (vocabulary.size**2,))
        self.assertAlmostEqual(float(distribution.sum()), 1.0)

    def test_jensen_shannon_is_zero_for_identical_inputs(self) -> None:
        distribution = np.array([0.2, 0.8], dtype=np.float64)
        self.assertAlmostEqual(
            jensen_shannon_divergence(distribution, distribution), 0.0
        )

    def test_jensen_shannon_is_symmetric_and_bounded(self) -> None:
        left = np.array([0.9, 0.1], dtype=np.float64)
        right = np.array([0.2, 0.8], dtype=np.float64)
        forward = jensen_shannon_divergence(left, right)
        reverse = jensen_shannon_divergence(right, left)
        self.assertAlmostEqual(forward, reverse)
        self.assertGreater(forward, 0.0)
        self.assertLessEqual(forward, math.log(2.0))

    def test_jensen_shannon_rejects_misaligned_or_negative_inputs(self) -> None:
        with self.assertRaises(CorpusShiftValidationError):
            jensen_shannon_divergence(
                np.array([1.0], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            )
        with self.assertRaises(CorpusShiftValidationError):
            jensen_shannon_divergence(
                np.array([1.0, -1.0], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            )

    def test_cross_record_transitions_expose_naive_concatenation(self) -> None:
        self.assertEqual(
            cross_record_transitions(("agra", "mumbai", "pune")),
            (("a", "m"), ("i", "p")),
        )

    def test_packed_stream_has_only_outer_boundaries(self) -> None:
        pairs = packed_stream_bigrams(("ab", "cd"))
        self.assertEqual(
            pairs, ((".", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "."))
        )

    def test_boundary_audit_counts_are_exact(self) -> None:
        audit = audit_boundaries(("ab", "cde", "f"))
        self.assertEqual(audit.record_count, 3)
        self.assertEqual(audit.character_count, 6)
        self.assertEqual(audit.boundary_aware_transitions, 9)
        self.assertEqual(audit.packed_transitions, 7)
        self.assertEqual(audit.cross_record_events, (("b", "c"), ("e", "f")))

    def test_experiment_has_complete_cross_evaluation_matrix(self) -> None:
        experiment = run_corpus_shift_experiment(BASELINE, SHIFTED, smoothing=0.5)
        pairs = {
            (evaluation.trained_on, evaluation.evaluated_on)
            for evaluation in experiment.evaluations
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

    def test_experiment_is_deterministic(self) -> None:
        first = run_corpus_shift_experiment(BASELINE, SHIFTED)
        second = run_corpus_shift_experiment(BASELINE, SHIFTED)
        self.assertEqual(experiment_metrics(first), experiment_metrics(second))
        np.testing.assert_array_equal(
            first.baseline_probabilities, second.baseline_probabilities
        )
        np.testing.assert_array_equal(
            first.shifted_probabilities, second.shifted_probabilities
        )

    def test_validation_rejects_incomplete_evaluations(self) -> None:
        experiment = run_corpus_shift_experiment(BASELINE, SHIFTED)
        corrupted = replace(experiment, evaluations=experiment.evaluations[:-1])
        with self.assertRaises(CorpusShiftValidationError):
            validate_experiment(corrupted)

    def test_validation_rejects_probability_and_boundary_tampering(self) -> None:
        experiment = run_corpus_shift_experiment(BASELINE, SHIFTED)
        probabilities = experiment.baseline_probabilities.copy()
        probabilities[0] = np.roll(probabilities[0], 1)
        with self.assertRaises(CorpusShiftValidationError):
            validate_experiment(
                replace(experiment, baseline_probabilities=probabilities)
            )
        with self.assertRaises(CorpusShiftValidationError):
            validate_experiment(
                replace(
                    experiment,
                    shifted_boundary_audit=experiment.baseline_boundary_audit,
                )
            )

    def test_checked_in_city_manifest_matches_corpus_bytes(self) -> None:
        corpus_path = ROOT / "data" / "day-21-indian-cities.txt"
        manifest = json.loads(
            (ROOT / "data" / "day-21-indian-cities.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        payload = corpus_path.read_bytes()
        words = corpus_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(manifest["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(manifest["word_count"], len(words))
        self.assertEqual(manifest["character_count"], sum(map(len, words)))
        self.assertEqual(manifest["unique_word_count"], len(set(words)))
        self.assertEqual(manifest["vocabulary"], [".", *sorted(set("".join(words)))])

    def test_metrics_report_shift_and_boundary_counts(self) -> None:
        experiment = run_corpus_shift_experiment(BASELINE, SHIFTED)
        metrics = experiment_metrics(experiment)
        self.assertEqual(metrics["baseline_records"], 3)
        self.assertEqual(metrics["shifted_records"], 3)
        self.assertEqual(metrics["baseline_cross_record_events"], 2)
        self.assertEqual(metrics["shifted_cross_record_events"], 2)
        self.assertGreater(metrics["transition_js_divergence"], 0.0)

    def test_payload_is_json_compatible_and_complete(self) -> None:
        payload = experiment_payload(run_corpus_shift_experiment(BASELINE, SHIFTED))
        self.assertEqual(len(payload["evaluations"]), 4)
        self.assertEqual(
            len(payload["boundary_audits"]["shifted"]["cross_record_events"]),
            2,
        )

    def test_mermaid_shows_positive_and_negative_controls(self) -> None:
        diagram = render_boundary_mermaid(
            run_corpus_shift_experiment(BASELINE, SHIFTED)
        )
        self.assertIn("boundary-aware encoding", diagram)
        self.assertIn("naive concatenation", diagram)

    def test_markdown_is_factual_about_scope(self) -> None:
        report = render_corpus_shift_markdown(
            run_corpus_shift_experiment(BASELINE, SHIFTED)
        )
        self.assertIn("Cross-corpus evaluation", report)
        self.assertIn("Boundary audit", report)
        self.assertIn("does not prove", report)
        self.assertIn("not a transformer-attention implementation", report)


if __name__ == "__main__":
    unittest.main()
