import copy
import math
import unittest

from ai_journey.sft_data import SFTDataError
from ai_journey.sft_experiment import SFTComparisonConfig, run_sft_comparison
from ai_journey.sft_report import summarize_sft_comparison


def result():
    return run_sft_comparison(
        SFTComparisonConfig(
            block_size=16,
            embedding_dim=8,
            head_count=2,
            batch_size=2,
            base_steps=1,
            sft_steps=1,
            evaluation_batch_size=64,
        )
    )


class SFTSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = result()

    def test_summary_reports_capacity_uncertainty_and_all_paired_deltas(self):
        summary = summarize_sft_comparison(self.result)
        self.assertEqual(
            [row["name"] for row in summary["arms"]],
            ["frozen", "full", "lora-rank-1", "lora-rank-2"],
        )
        self.assertEqual(len(summary["paired"]), 6)
        self.assertEqual(summary["arms"][0]["trainable_fraction_of_full"], 0)
        self.assertEqual(summary["arms"][1]["trainable_fraction_of_full"], 1)
        self.assertLess(summary["arms"][2]["trainable_fraction_of_full"], 1)
        for arm in summary["arms"]:
            stats = arm["assistant_mean_nll"]
            self.assertEqual(stats["count"], 2)
            self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertIn("not confidence intervals", summary["uncertainty_policy"])

    def test_paired_delta_is_candidate_minus_reference_with_seed_identity(self):
        summary = summarize_sft_comparison(self.result)
        comparison = next(
            row
            for row in summary["paired"]
            if row["reference"] == "full" and row["candidate"] == "lora-rank-1"
        )
        expected = []
        for seed in self.result["seeds"]:
            scores = {
                trial["name"]: trial["heldout"]["mean_nll"] for trial in seed["trials"]
            }
            expected.append(scores["lora-rank-1"] - scores["full"])
        self.assertEqual([row["seed"] for row in comparison["per_seed"]], [43, 44])
        self.assertEqual([row["delta_nll"] for row in comparison["per_seed"]], expected)

    def test_rejects_missing_arms_coverage_nonfinite_and_parameter_tampering(self):
        mutations = []
        missing = copy.deepcopy(self.result)
        missing["seeds"][0]["trials"].pop()
        mutations.append(missing)
        coverage = copy.deepcopy(self.result)
        coverage["seeds"][0]["trials"][0]["heldout"]["target_tokens"] -= 1
        mutations.append(coverage)
        nonfinite = copy.deepcopy(self.result)
        nonfinite["seeds"][0]["trials"][0]["heldout"]["mean_nll"] = float("nan")
        mutations.append(nonfinite)
        counts = copy.deepcopy(self.result)
        counts["seeds"][0]["trials"][1]["parameters"]["total"] += 1
        mutations.append(counts)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(SFTDataError):
                summarize_sft_comparison(mutation)


if __name__ == "__main__":
    unittest.main()
