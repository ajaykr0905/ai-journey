import copy
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.lora_data import LoRAExperimentError
from ai_journey.lora_experiment import COMPARISON_ARMS, ComparisonConfig, run_comparison
from ai_journey.lora_report import (
    build_report,
    render_report,
    report_digest,
    summarize_comparison,
    validate_report,
    write_report,
)


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


class ComparisonReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_comparison(
            ComparisonConfig(
                block_size=3,
                embedding_dim=4,
                head_count=1,
                batch_size=2,
                pretrain_steps=1,
                finetune_steps=1,
            )
        )
        cls.report = build_report(cls.result)

    def test_canonical_report_roundtrip_is_deterministic_and_owns_nested_data(self):
        result = copy.deepcopy(self.result)
        first = build_report(result)
        second = build_report(self.result)
        self.assertEqual(render_report(first), render_report(second))
        result["seeds"][0]["trials"][0]["training"]["losses"][0] += 1
        self.assertEqual(first, second)
        validate_report(json.loads(render_report(first)))
        self.assertEqual(first["sha256"], report_digest(first))

    def test_rehashed_semantic_tampering_is_rejected_without_retraining(self):
        mutations = (
            lambda r: r.update(extra=True),
            lambda r: r["result"]["fixture"]["train"]["document_ids"].reverse(),
            lambda r: r["result"]["heldout"].update(target_tokens=375),
            lambda r: r["result"]["model"].update(head_count=True),
            lambda r: r["result"]["runtime"].update(threads=True),
            lambda r: r["result"]["config"].update(seeds=[11]),
            lambda r: r["result"]["config"].update(
                block_size=32,
                embedding_dim=32,
                batch_size=8,
                pretrain_steps=64,
                finetune_steps=64,
            ),
            lambda r: r["result"]["seeds"][0].update(source_after_sha256="a" * 64),
            lambda r: r["result"]["seeds"][0]["adaptation_schedule"].update(
                sha256="a" * 64
            ),
            lambda r: r["result"]["seeds"][0]["trials"][2]["parameters"].update(
                trainable=25
            ),
            lambda r: r["result"]["seeds"][0]["trials"][2]["adapter"].update(alpha=9),
            lambda r: r["result"]["seeds"][0]["trials"][2].update(
                frozen_after_sha256="a" * 64
            ),
            lambda r: r["result"]["seeds"][0]["trials"][2]["merge"].update(
                probe_max_abs_error=1e-5
            ),
            lambda r: r["result"]["seeds"][0]["trials"][0]["training"].update(
                optimizer_steps=1
            ),
            lambda r: r["result"]["seeds"][0]["trials"][1]["training"]["losses"].pop(),
            lambda r: r["result"]["seeds"][0]["trials"][1]["heldout"].update(
                mean_nll=1.0
            ),
            lambda r: r["summary"]["paired"][0].update(candidate_wins=99),
        )
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            report["sha256"] = report_digest(report)
            with self.subTest(mutate=mutate), self.assertRaises(LoRAExperimentError):
                validate_report(report)

    def test_invalid_digest_nonfinite_and_missing_fields_are_errors(self):
        for value in (None, [], {}, {"schema": "other"}):
            with self.subTest(value=value), self.assertRaises(LoRAExperimentError):
                validate_report(value)
        report = copy.deepcopy(self.report)
        report["result"]["seeds"][0]["trials"][0]["heldout"]["mean_nll"] = float("nan")
        with self.assertRaises(LoRAExperimentError):
            validate_report(report)
        report = copy.deepcopy(self.report)
        report["sha256"] = "0" * 64
        with self.assertRaisesRegex(LoRAExperimentError, "digest"):
            validate_report(report)

    def test_atomic_publication_preserves_previous_report_on_validation_or_replace_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            write_report(path, self.report)
            before = path.read_bytes()
            invalid = copy.deepcopy(self.report)
            invalid["summary"]["paired"].pop()
            with self.assertRaises(LoRAExperimentError):
                write_report(path, invalid)
            self.assertEqual(path.read_bytes(), before)
            with patch(
                "ai_journey.lora_report.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaises(OSError):
                    write_report(path, self.report)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(json.loads(before), self.report)
