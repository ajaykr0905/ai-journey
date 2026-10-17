import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_journey.sft_data import SFTDataError
from ai_journey.sft_experiment import SFTComparisonConfig, run_sft_comparison
from ai_journey.sft_report import (
    build_sft_report,
    render_sft_report,
    sft_report_digest,
    summarize_sft_comparison,
    validate_sft_report,
    write_sft_report,
)


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


class SFTReportValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_sft_report(result())

    def test_builds_self_verifying_report_with_recomputed_summary(self):
        self.assertIs(validate_sft_report(self.report), self.report)
        self.assertEqual(self.report["sha256"], sft_report_digest(self.report))
        self.assertEqual(self.report["schema"], "ai-journey-chat-sft-comparison-v1")

    def test_rejects_digest_schedule_metric_summary_and_fixture_tampering(self):
        mutations = []
        digest = copy.deepcopy(self.report)
        digest["sha256"] = "0" * 64
        mutations.append(digest)
        schedule = copy.deepcopy(self.report)
        schedule["result"]["seeds"][0]["sft_schedule"]["input_tokens"] += 1
        schedule["sha256"] = sft_report_digest(schedule)
        mutations.append(schedule)
        metric = copy.deepcopy(self.report)
        metric["result"]["seeds"][0]["trials"][0]["heldout"]["mean_nll"] += 1
        metric["sha256"] = sft_report_digest(metric)
        mutations.append(metric)
        summary = copy.deepcopy(self.report)
        summary["summary"]["arms"][0]["assistant_mean_nll"]["mean"] += 1
        summary["sha256"] = sft_report_digest(summary)
        mutations.append(summary)
        fixture = copy.deepcopy(self.report)
        fixture["result"]["fixture"]["train"]["characters"] += 1
        fixture["sha256"] = sft_report_digest(fixture)
        mutations.append(fixture)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(SFTDataError):
                validate_sft_report(mutation)


class SFTReportPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_sft_report(result())

    def test_writes_repeatable_canonical_json_to_nested_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            write_sft_report(path, self.report)
            before = path.read_bytes()
            write_sft_report(path, self.report)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(json.loads(before), self.report)
            self.assertEqual(before.decode(), render_sft_report(self.report))
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp-*")), [])

    def test_invalid_report_and_replace_failure_preserve_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(b"previous")
            invalid = copy.deepcopy(self.report)
            invalid["sha256"] = "0" * 64
            with self.assertRaises(SFTDataError):
                write_sft_report(path, invalid)
            self.assertEqual(path.read_bytes(), b"previous")
            with (
                patch("ai_journey.sft_report.os.replace", side_effect=OSError("stop")),
                self.assertRaises(OSError),
            ):
                write_sft_report(path, self.report)
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(path.parent.glob(f".{path.name}.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
