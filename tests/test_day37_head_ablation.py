from dataclasses import asdict
from pathlib import Path
import unittest

import torch

from ai_journey.ablation_protocol import (
    build_ablation_report,
    evaluate_ablation,
    load_ablation_protocol,
    run_ablation,
    verify_ablation_report,
)
from ai_journey.transformer_lab import TokenCorpus

ROOT = Path(__file__).resolve().parents[1]


class HeadCountAblationTests(unittest.TestCase):
    def test_three_arms_change_only_head_count(self):
        protocol = load_ablation_protocol(
            ROOT / "config/day-37-head-count-ablation.json", vocab_size=12
        )
        baseline = asdict(protocol.baseline_arm.model_config)
        self.assertEqual(protocol.trial_seeds, (37, 38))
        self.assertEqual(
            [arm.model_config.head_count for arm in protocol.arms], [1, 2, 4]
        )
        for arm in protocol.arms:
            values = asdict(arm.model_config)
            values["head_count"] = baseline["head_count"]
            self.assertEqual(values, baseline)
            self.assertEqual(
                asdict(arm.training_config),
                asdict(protocol.baseline_arm.training_config),
            )

    def test_matched_cpu_trials_are_reproducible_and_report_negative_outcomes(self):
        corpus = TokenCorpus.from_path(
            ROOT / "data/day-19-demo-names.txt", block_size=4
        )
        protocol = load_ablation_protocol(
            ROOT / "config/day-37-head-count-ablation.json",
            vocab_size=corpus.vocab_size,
        )
        rng = torch.get_rng_state().clone()
        first = run_ablation(corpus, protocol)
        second = run_ablation(corpus, protocol)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        evaluation = evaluate_ablation(first)
        self.assertTrue(evaluation.complete_pairs)
        self.assertTrue(evaluation.matched_first_batches)
        # A head-count change preserves the packed parameter shapes and RNG draws.
        for seed in protocol.trial_seeds:
            self.assertEqual(
                len(
                    {
                        trial.initial_model_fingerprint
                        for trial in first.trials
                        if trial.seed == seed
                    }
                ),
                1,
            )
        self.assertEqual(len(first.trials), 6)
        self.assertIn(
            evaluation.outcome, {"supported", "rejected", "inconclusive", "mixed"}
        )
        verify_ablation_report(build_ablation_report(first))
