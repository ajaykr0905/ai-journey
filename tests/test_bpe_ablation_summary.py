from copy import deepcopy
import unittest

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_ablation_summary import summarize_ablation
from ai_journey.bpe_model_ablation import VocabularyProtocol, run_vocabulary_ablation


class PairedAblationSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_vocabulary_ablation(
            ("banana banana banana", "abracadabra abracadabra"),
            ("held out", "नमस्ते"),
            VocabularyProtocol(
                vocab_sizes=(256, 258, 260), seeds=(1, 2), steps=1, block_size=3
            ),
        )

    def test_paired_delta_uses_same_seed_baseline_and_keeps_negative_findings(self):
        summary = summarize_ablation(self.report)
        self.assertEqual(summary["arms"][0]["paired_deltas_vs_byte"], [0, 0])
        for arm in summary["arms"]:
            rows = [
                row
                for row in self.report["trials"]
                if row["requested_vocab_size"] == arm["requested_vocab_size"]
            ]
            baseline = [
                row
                for row in self.report["trials"]
                if row["requested_vocab_size"] == 256
            ]
            expected = [
                left["final"]["content_bits_per_byte"]
                - right["final"]["content_bits_per_byte"]
                for left, right in zip(rows, baseline)
            ]
            self.assertEqual(arm["paired_deltas_vs_byte"], expected)
            self.assertAlmostEqual(arm["mean_paired_delta"], sum(expected) / 2)

    def test_tampering_or_incomplete_pairing_is_rejected(self):
        mutations = [
            lambda report: report["trials"].pop(),
            lambda report: report["trials"].append(deepcopy(report["trials"][0])),
            lambda report: report["trials"][0]["final"].update(utf8_bytes=1),
            lambda report: report["trials"][0].update(parameter_count=1),
            lambda report: report["trials"][0].update(
                initial_trunk_fingerprint="0" * 64
            ),
            lambda report: report["trials"][0]["trace"][0].update(loss=float("nan")),
            lambda report: report["protocol"].update(steps=100),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                report = deepcopy(self.report)
                mutate(report)
                with self.assertRaises(TokenizerError):
                    summarize_ablation(report)
