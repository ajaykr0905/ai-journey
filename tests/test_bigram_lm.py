from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.bigram_lm import (
    BOUNDARY_TOKEN,
    DEFAULT_CORPUS,
    BigramModel,
    BigramValidationError,
    Vocabulary,
    build_bigram_model,
    build_manifest,
    build_vocabulary,
    canonical_corpus_bytes,
    count_bigrams,
    experiment_metrics,
    experiment_payload,
    iter_bigrams,
    load_corpus,
    loop_probability_oracle,
    manifest_payload,
    normalize_corpus,
    normalize_counts,
    render_bigram_markdown,
    render_bigram_mermaid,
    run_bigram_experiment,
    sample_word,
    sample_words,
    top_transitions,
    transition_probability,
    validate_experiment,
    validate_model,
)

CORPUS_PATH = ROOT / "data" / "day-19-demo-names.txt"
MANIFEST_PATH = ROOT / "data" / "day-19-demo-names.manifest.json"


class CorpusAndVocabularyTests(unittest.TestCase):
    def test_checked_in_corpus_matches_library_default(self) -> None:
        self.assertEqual(load_corpus(CORPUS_PATH), DEFAULT_CORPUS)

    def test_normalize_corpus_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(normalize_corpus(["  alara ", "belin\n"]), ("alara", "belin"))

    def test_normalize_corpus_rejects_empty_non_string_and_single_string(self) -> None:
        with self.assertRaises(BigramValidationError):
            normalize_corpus([])
        with self.assertRaises(BigramValidationError):
            normalize_corpus([" "])
        with self.assertRaises(TypeError):
            normalize_corpus(["valid", 7])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            normalize_corpus("oneword")

    def test_normalize_corpus_enforces_public_format_contract(self) -> None:
        for corpus in (["Upper"], ["two words"], ["na.me"], ["café"]):
            with self.subTest(corpus=corpus), self.assertRaises(BigramValidationError):
                normalize_corpus(corpus)

    def test_canonical_bytes_use_lf_and_final_newline(self) -> None:
        self.assertEqual(canonical_corpus_bytes(["ada", "bea"]), b"ada\nbea\n")

    def test_vocabulary_is_boundary_first_sorted_and_unique(self) -> None:
        vocabulary = build_vocabulary(["cab", "bed"])
        self.assertEqual(vocabulary.tokens, (".", "a", "b", "c", "d", "e"))
        self.assertEqual(vocabulary.size, 6)

    def test_vocabulary_encode_decode_round_trip_and_errors(self) -> None:
        vocabulary = build_vocabulary(["ada"])
        for index, token in enumerate(vocabulary.tokens):
            self.assertEqual(vocabulary.encode(token), index)
            self.assertEqual(vocabulary.decode(index), token)
        with self.assertRaises(BigramValidationError):
            vocabulary.encode("z")
        with self.assertRaises(BigramValidationError):
            vocabulary.decode(vocabulary.size)
        with self.assertRaises(TypeError):
            vocabulary.decode(True)

    def test_vocabulary_requires_boundary_at_zero_and_unique_single_characters(
        self,
    ) -> None:
        for tokens in (("a",), (".", "a", "a"), (".", "ab")):
            with self.subTest(tokens=tokens), self.assertRaises(BigramValidationError):
                Vocabulary(tokens)

    def test_iter_bigrams_adds_both_boundaries(self) -> None:
        self.assertEqual(
            iter_bigrams("ada"), ((".", "a"), ("a", "d"), ("d", "a"), ("a", "."))
        )


class CountAndProbabilityTests(unittest.TestCase):
    def test_count_bigrams_matches_hand_counted_tiny_corpus(self) -> None:
        words = ("ab", "aa")
        vocabulary = build_vocabulary(words)
        counts = count_bigrams(words, vocabulary)
        dot, a, b = (vocabulary.encode(token) for token in ".ab")
        self.assertEqual(counts[dot, a], 2)
        self.assertEqual(counts[a, a], 1)
        self.assertEqual(counts[a, b], 1)
        self.assertEqual(counts[a, dot], 1)
        self.assertEqual(counts[b, dot], 1)
        self.assertEqual(int(counts.sum()), 6)

    def test_transition_total_includes_one_end_boundary_per_word(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        self.assertEqual(
            int(model.counts.sum()), sum(len(word) + 1 for word in DEFAULT_CORPUS)
        )

    def test_normalize_counts_broadcasts_column_totals_over_rows(self) -> None:
        counts = np.array([[1, 3], [2, 2]], dtype=np.int64)
        probabilities = normalize_counts(counts)
        np.testing.assert_allclose(probabilities, [[0.25, 0.75], [0.5, 0.5]])
        self.assertEqual(counts.sum(axis=1, keepdims=True).shape, (2, 1))

    def test_normalize_counts_rejects_wrong_shape_dtype_values_and_empty_rows(
        self,
    ) -> None:
        invalid = (
            np.array([1, 2]),
            np.ones((2, 3), dtype=np.int64),
            np.ones((2, 2), dtype=np.float64),
            np.array([[1, -1], [1, 1]], dtype=np.int64),
            np.array([[1, 1], [0, 0]], dtype=np.int64),
        )
        for counts in invalid:
            with (
                self.subTest(counts=counts),
                self.assertRaises((BigramValidationError, TypeError)),
            ):
                normalize_counts(counts)

    def test_vectorized_probabilities_match_independent_loop_oracle(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        np.testing.assert_array_equal(
            model.probabilities, loop_probability_oracle(model.counts)
        )

    def test_model_probability_rows_are_finite_nonnegative_and_normalized(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        validate_model(model)
        self.assertTrue(np.all(np.isfinite(model.probabilities)))
        self.assertTrue(np.all(model.probabilities >= 0.0))
        np.testing.assert_allclose(model.probabilities.sum(axis=1), 1.0)

    def test_validate_model_detects_shape_and_probability_corruption(self) -> None:
        model = build_bigram_model(["ada", "ava"])
        with self.assertRaises(BigramValidationError):
            validate_model(replace(model, counts=model.counts[:2, :2]))
        corrupt = model.probabilities.copy()
        corrupt[0, :] = 0.0
        with self.assertRaises(BigramValidationError):
            validate_model(replace(model, probabilities=corrupt))

    def test_transition_probability_uses_token_labels(self) -> None:
        model = build_bigram_model(["ab", "aa"])
        self.assertEqual(transition_probability(model, ".", "a"), 1.0)
        self.assertEqual(transition_probability(model, "b", "."), 1.0)
        self.assertAlmostEqual(transition_probability(model, "a", "."), 1 / 3)

    def test_top_transitions_are_observed_ranked_and_limited(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        transitions = top_transitions(model, 7)
        self.assertEqual(len(transitions), 7)
        self.assertTrue(all(transition.count > 0 for transition in transitions))
        self.assertEqual(
            list(transitions),
            sorted(
                transitions, key=lambda item: (-item.count, item.previous, item.next)
            ),
        )
        with self.assertRaises(BigramValidationError):
            top_transitions(model, 0)


class SamplingTests(unittest.TestCase):
    def test_sample_sequences_are_reproducible_for_same_seed(self) -> None:
        model = build_bigram_model(DEFAULT_CORPUS)
        self.assertEqual(sample_words(model, seed=11), sample_words(model, seed=11))
        self.assertNotEqual(sample_words(model, seed=11), sample_words(model, seed=12))

    def test_default_smoke_samples_all_terminate_and_reconcile(self) -> None:
        experiment = run_bigram_experiment()
        self.assertTrue(all(sample.terminated for sample in experiment.samples))
        for sample in experiment.samples:
            self.assertEqual(
                len(sample.token_ids), len(sample.transition_probabilities) + 1
            )
            self.assertEqual(sample.token_ids[0], 0)
            self.assertEqual(sample.token_ids[-1], 0)

    def test_sample_word_reports_max_length_without_false_termination(self) -> None:
        vocabulary = Vocabulary((BOUNDARY_TOKEN, "a"))
        counts = np.array([[0, 1], [0, 1]], dtype=np.int64)
        model = BigramModel(vocabulary, counts, normalize_counts(counts))
        sample = sample_word(model, np.random.default_rng(1), max_length=3)
        self.assertEqual(sample.text, "aaa")
        self.assertFalse(sample.terminated)
        self.assertEqual(sample.token_ids, (0, 1, 1, 1))

    def test_sampling_rejects_invalid_count_seed_and_max_length(self) -> None:
        model = build_bigram_model(["ada"])
        with self.assertRaises(BigramValidationError):
            sample_words(model, count=0)
        with self.assertRaises(TypeError):
            sample_words(model, seed=True)
        with self.assertRaises(BigramValidationError):
            sample_word(model, np.random.default_rng(1), max_length=0)


class ManifestExperimentAndReportTests(unittest.TestCase):
    def test_checked_in_manifest_matches_computed_manifest(self) -> None:
        expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        actual = manifest_payload(build_manifest(load_corpus(CORPUS_PATH)))
        self.assertEqual(actual, expected)

    def test_manifest_hash_is_stable_and_counts_reconcile(self) -> None:
        manifest = build_manifest(DEFAULT_CORPUS)
        self.assertEqual(
            manifest.sha256,
            "f3670d269756a7b8800389c29599c25b2eccc63ca16e7b4f84b532d21f973411",
        )
        self.assertEqual(manifest.word_count, 20)
        self.assertEqual(manifest.character_count, 100)
        self.assertEqual(manifest.unique_word_count, 20)

    def test_experiment_metrics_capture_shapes_oracle_and_smoke_status(self) -> None:
        metrics = experiment_metrics(run_bigram_experiment())
        self.assertEqual(metrics["probability_shape"], "18x18")
        self.assertEqual(metrics["row_total_shape"], "18x1")
        self.assertEqual(metrics["observed_transitions"], 120)
        self.assertEqual(metrics["nonzero_bigram_types"], 58)
        self.assertEqual(metrics["oracle_max_error"], 0.0)
        self.assertEqual(metrics["terminated_samples"], 10)

    def test_experiment_payload_is_json_serializable_and_complete(self) -> None:
        payload = experiment_payload(run_bigram_experiment(sample_count=3))
        encoded = json.dumps(payload)
        self.assertIn("counts", payload)
        self.assertIn("probabilities", payload)
        self.assertEqual(len(payload["samples"]), 3)  # type: ignore[arg-type]
        self.assertIn("corpus_sha256", encoded)

    def test_validate_experiment_detects_manifest_and_oracle_tampering(self) -> None:
        experiment = run_bigram_experiment()
        with self.assertRaises(BigramValidationError):
            validate_experiment(replace(experiment, oracle_max_error=0.1))
        bad_manifest = replace(experiment.manifest, word_count=999)
        with self.assertRaises(BigramValidationError):
            validate_experiment(replace(experiment, manifest=bad_manifest))

    def test_markdown_report_contains_manifest_broadcast_samples_and_disclaimer(
        self,
    ) -> None:
        report = render_bigram_markdown(run_bigram_experiment(sample_count=3))
        self.assertIn("Data manifest", report)
        self.assertIn("Broadcast row totals: `18x1`", report)
        self.assertIn("Maximum broadcast/oracle error: 0.000e+00", report)
        self.assertIn("does not prove", report)
        self.assertIn("```mermaid", report)

    def test_mermaid_report_contains_ranked_transition_evidence(self) -> None:
        graph = render_bigram_mermaid(build_bigram_model(DEFAULT_CORPUS), limit=4)
        self.assertTrue(graph.startswith("flowchart LR\n"))
        self.assertEqual(graph.count("-->|"), 4)
        self.assertIn("START_END", graph)


if __name__ == "__main__":
    unittest.main()
