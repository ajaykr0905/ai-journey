import copy
import math
import unittest

from ai_journey.lora_data import LoRAExperimentError
from ai_journey.lora_experiment import COMPARISON_ARMS
from ai_journey.lora_report import summarize_comparison


def measurement_fixture():
    scores = ((4.0, 3.0, 2.0, 4.0), (5.0, 2.0, 6.0, 3.0))
    return {
        "config": {"seeds": [11, 29]},
        "heldout": {"target_tokens": 17},
        "seeds": [
            {
                "seed": seed,
                "trials": [
                    {
                        "name": arm,
                        "heldout": {"mean_nll": nll, "target_tokens": 17},
                        "parameters": {
                            "total": total,
                            "trainable": trainable,
                            "frozen": total - trainable,
                        },
                    }
                    for arm, nll, total, trainable in zip(
                        COMPARISON_ARMS,
                        values,
                        (100, 100, 110, 120),
                        (0, 100, 10, 20),
                        strict=True,
                    )
                ],
            }
            for seed, values in zip((11, 29), scores, strict=True)
        ],
    }


class PairedSummaryTests(unittest.TestCase):
    def test_paired_uncertainty_uses_differences_before_aggregation(self):
        result = measurement_fixture()
        before = copy.deepcopy(result)
        summary = summarize_comparison(result)
        self.assertEqual(result, before)
        self.assertEqual(len(summary["paired"]), 6)
        full = summary["paired"][0]
        self.assertEqual((full["reference"], full["candidate"]), ("frozen", "full"))
        self.assertEqual(
            full["per_seed"],
            [{"seed": 11, "delta_nll": -1}, {"seed": 29, "delta_nll": -3}],
        )
        self.assertEqual(full["delta_nll"]["mean"], -2)
        self.assertAlmostEqual(full["delta_nll"]["sample_stddev"], math.sqrt(2))
        self.assertEqual(full["delta_nll"]["standard_error"], 1)
        self.assertEqual(full["candidate_wins"], 2)
        self.assertEqual(summary["paired"][1]["candidate_wins"], 1)
        self.assertEqual(summary["paired"][2]["ties"], 1)
        self.assertEqual(summary["arms"][2]["trainable_fraction_of_full"], 0.1)
        self.assertIn("not confidence", summary["uncertainty_policy"])

    def test_zero_spread_and_ties_are_reported_without_a_winner_claim(self):
        result = measurement_fixture()
        for row in result["seeds"]:
            for trial in row["trials"]:
                trial["heldout"]["mean_nll"] = 3.0
        for row in summarize_comparison(result)["paired"]:
            self.assertEqual(row["delta_nll"]["sample_stddev"], 0)
            self.assertEqual(row["candidate_wins"], 0)
            self.assertEqual(row["ties"], 2)

    def test_rejects_incomplete_pairing_nonfinite_scores_and_mismatched_coverage(self):
        mutations = (
            lambda r: r["seeds"].pop(),
            lambda r: r["seeds"][0]["trials"].pop(),
            lambda r: r["seeds"][0]["trials"].reverse(),
            lambda r: r["seeds"][0]["trials"][0]["heldout"].update(
                mean_nll=float("nan")
            ),
            lambda r: r["seeds"][0]["trials"][0]["heldout"].update(target_tokens=18),
            lambda r: r["seeds"][1]["trials"][1]["parameters"].update(
                total=101, frozen=1
            ),
            lambda r: r["seeds"][0]["trials"][0]["parameters"].update(trainable=True),
        )
        for mutate in mutations:
            result = measurement_fixture()
            mutate(result)
            with self.subTest(mutate=mutate), self.assertRaises(LoRAExperimentError):
                summarize_comparison(result)
