from __future__ import annotations

import json
import math
import random
import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.initialization_comparison import (
    ComparisonCriteria,
    audit_model_initialization,
    build_comparison_report,
    evaluate_comparison,
    render_initialization_audit,
    render_loss_curves,
    run_kaiming_comparison,
    summarize_loss_curve,
    write_comparison_report,
)
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    StepMetric,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
    seed_everything,
)


class InitializationAuditTests(unittest.TestCase):
    def test_public_package_exports_comparison_entry_points(self) -> None:
        import ai_journey

        self.assertIs(ai_journey.run_kaiming_comparison, run_kaiming_comparison)
        self.assertIs(ai_journey.ComparisonCriteria, ComparisonCriteria)

    def test_audit_covers_every_initialized_matrix_in_module_order(self) -> None:
        seed_everything(26)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=17,
                block_size=4,
                embedding_dim=16,
                head_count=4,
                layer_count=1,
            )
        )
        audit = audit_model_initialization(model)
        expected_names = tuple(
            f"{name}.weight"
            for name, module in model.named_modules()
            if module.__class__.__name__ in {"Linear", "Embedding"}
        )
        self.assertEqual(tuple(item.name for item in audit), expected_names)
        self.assertTrue(all(item.observed_std > 0 for item in audit))
        self.assertTrue(
            all(item.bias_is_zero is not False for item in audit),
            audit,
        )

    def test_kaiming_audit_records_fan_in_gain_and_fixed_embeddings(self) -> None:
        gain = 1.25
        config = TransformerConfig(
            vocab_size=17,
            block_size=4,
            embedding_dim=16,
            head_count=4,
            layer_count=1,
            initialization_mode="kaiming_normal",
            initialization_gain=gain,
            initialization_std=0.03,
        )
        seed_everything(26)
        audit = audit_model_initialization(DecoderLanguageModel(config))
        by_name = {item.name: item for item in audit}
        linear = by_name["blocks.0.feed_forward.network.0.weight"]
        embedding = by_name["token_embedding.weight"]
        self.assertEqual(linear.fan_in, config.embedding_dim)
        self.assertEqual(
            linear.expected_std,
            gain / math.sqrt(config.embedding_dim),
        )
        self.assertIsNone(embedding.fan_in)
        self.assertEqual(embedding.expected_std, config.initialization_std)
        self.assertIsNone(embedding.bias_is_zero)

    def test_audit_rejects_unrelated_module(self) -> None:
        from torch import nn

        with self.assertRaisesRegex(TypeError, "DecoderLanguageModel"):
            audit_model_initialization(nn.Linear(2, 2))  # type: ignore[arg-type]


class KaimingComparisonTests(unittest.TestCase):
    def _corpus(self, directory: str) -> TokenCorpus:
        path = Path(directory) / "corpus.txt"
        path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
        return TokenCorpus.from_path(path, block_size=4)

    def test_comparison_is_deterministic_and_changes_only_policy(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            model_config = TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                initialization_gain=1.0,
            )
            training_config = TrainingConfig(
                steps=2,
                batch_size=4,
                learning_rate=0.01,
                seed=26,
            )
            first = run_kaiming_comparison(
                corpus,
                model_config=model_config,
                training_config=training_config,
            )
            second = run_kaiming_comparison(
                corpus,
                model_config=model_config,
                training_config=training_config,
            )
        self.assertEqual(first.to_dict(), second.to_dict())
        payload = first.to_dict()
        fingerprint = payload.pop("evidence_fingerprint")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            fingerprint,
            sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        self.assertEqual(fingerprint, first.evidence_fingerprint())
        self.assertEqual(first.runtime.device, "cpu")
        self.assertTrue(first.runtime.python_version)
        self.assertTrue(first.runtime.torch_version)
        self.assertTrue(first.runtime.numpy_version)
        self.assertTrue(first.runtime.machine)
        self.assertTrue(first.runtime.deterministic_algorithms)
        self.assertGreater(first.runtime.intraop_threads, 0)
        fixed, kaiming = first.variants
        self.assertEqual((fixed.name, kaiming.name), ("fixed_normal", "kaiming_normal"))
        fixed_config = fixed.to_dict()["model_config"]
        kaiming_config = kaiming.to_dict()["model_config"]
        differing = {
            key for key in fixed_config if fixed_config[key] != kaiming_config[key]
        }
        self.assertEqual(differing, {"initialization_mode"})
        self.assertEqual(len(fixed.trace), training_config.steps)
        self.assertEqual(len(kaiming.trace), training_config.steps)
        self.assertNotEqual(
            [item.loss for item in fixed.trace],
            [item.loss for item in kaiming.trace],
        )
        self.assertEqual(fixed.loss_curve.step_count, training_config.steps)
        self.assertGreater(first.contrast.kaiming_to_fixed_mean_loss_ratio, 0)
        self.assertEqual(
            first.contrast.final_train_nll_delta,
            kaiming.final_train_nll - fixed.final_train_nll,
        )
        evaluation = evaluate_comparison(
            first,
            ComparisonCriteria(
                max_relative_std_error=1.0,
                min_variant_loss_reduction=0.0,
                min_kaiming_mean_loss_improvement=0.0,
            ),
        )
        self.assertTrue(evaluation.passed, evaluation.violations)

    def test_comparison_rejects_invalid_baseline_or_vocabulary(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            with self.assertRaisesRegex(TransformerLabError, "fixed_normal"):
                run_kaiming_comparison(
                    corpus,
                    model_config=TransformerConfig(
                        vocab_size=corpus.vocab_size,
                        initialization_mode="kaiming_normal",
                    ),
                    training_config=TrainingConfig(steps=1),
                )
            with self.assertRaisesRegex(TransformerLabError, "vocabulary"):
                run_kaiming_comparison(
                    corpus,
                    model_config=TransformerConfig(vocab_size=corpus.vocab_size + 1),
                    training_config=TrainingConfig(steps=1),
                )

    def test_comparison_restores_caller_random_state(self) -> None:
        import tempfile

        import numpy as np
        import torch

        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        torch.use_deterministic_algorithms(False)
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            run_kaiming_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
            )
        actual = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_loss_curve_summary_reports_improvement_and_best_step(self) -> None:
        trace = (
            StepMetric(step=2, loss=4.0, gradient_norm=1.0),
            StepMetric(step=3, loss=2.0, gradient_norm=1.0),
            StepMetric(step=4, loss=3.0, gradient_norm=1.0),
        )
        summary = summarize_loss_curve(trace)
        self.assertEqual(summary.step_count, 3)
        self.assertEqual(summary.best_loss, 2.0)
        self.assertEqual(summary.best_step, 3)
        self.assertEqual(summary.mean_loss, 3.0)
        self.assertEqual(summary.relative_loss_reduction, 0.25)
        self.assertEqual(summary.improving_transition_fraction, 0.5)

    def test_loss_curve_summary_rejects_invalid_traces(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "must not be empty"):
            summarize_loss_curve(())
        with self.assertRaisesRegex(TransformerLabError, "finite"):
            summarize_loss_curve(
                (StepMetric(step=0, loss=float("nan"), gradient_norm=1.0),)
            )
        with self.assertRaisesRegex(TransformerLabError, "strictly increasing"):
            summarize_loss_curve(
                (
                    StepMetric(step=1, loss=1.0, gradient_norm=1.0),
                    StepMetric(step=1, loss=0.5, gradient_norm=1.0),
                )
            )

    def test_evaluation_reports_named_threshold_violations(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_kaiming_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    initialization_gain=1.0,
                ),
                training_config=TrainingConfig(
                    steps=2,
                    batch_size=4,
                    learning_rate=0.01,
                    seed=26,
                ),
            )
        evaluation = evaluate_comparison(
            result,
            ComparisonCriteria(
                max_relative_std_error=0,
                min_variant_loss_reduction=0.99,
                min_kaiming_mean_loss_improvement=0.99,
            ),
        )
        self.assertFalse(evaluation.passed)
        self.assertEqual(
            set(evaluation.violations),
            {
                "initialization_std_error",
                "insufficient_variant_loss_reduction",
                "insufficient_kaiming_mean_loss_improvement",
            },
        )

    def test_comparison_criteria_reject_invalid_thresholds(self) -> None:
        invalid = (
            {"max_relative_std_error": -1},
            {"min_variant_loss_reduction": 1},
            {"min_kaiming_mean_loss_improvement": float("nan")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(TransformerLabError):
                ComparisonCriteria(**values)

    def test_report_writer_is_stable_atomic_and_self_verifying(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_kaiming_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    initialization_gain=1.0,
                ),
                training_config=TrainingConfig(
                    steps=2,
                    batch_size=4,
                    learning_rate=0.01,
                    seed=26,
                ),
            )
            criteria = ComparisonCriteria(
                max_relative_std_error=1.0,
                min_variant_loss_reduction=0,
                min_kaiming_mean_loss_improvement=0,
            )
            output = Path(directory) / "nested" / "day-26.json"
            write_comparison_report(output, result, criteria)
            first = output.read_bytes()
            write_comparison_report(output, result, criteria)
            second = output.read_bytes()
            payload = json.loads(second)
        self.assertEqual(first, second)
        self.assertFalse(output.with_name(f".{output.name}.tmp").exists())
        fingerprint = payload.pop("report_fingerprint")
        expected = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(fingerprint, expected)
        self.assertEqual(
            build_comparison_report(result, criteria)["report_fingerprint"], expected
        )
        self.assertTrue(payload["evaluation"]["passed"])

    def test_loss_curve_renderer_writes_stable_atomic_svg(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_kaiming_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    initialization_gain=1.0,
                ),
                training_config=TrainingConfig(
                    steps=2,
                    batch_size=4,
                    learning_rate=0.01,
                    seed=26,
                ),
            )
            output = Path(directory) / "plots" / "loss-curves.svg"
            render_loss_curves(output, result)
            first = output.read_bytes()
            render_loss_curves(output, result)
            second = output.read_bytes()
        self.assertEqual(first, second)
        self.assertFalse(output.with_name(f".{output.name}.tmp").exists())
        self.assertIn(b"Matched transformer loss curves", first)
        self.assertIn(b"fixed_normal", first)
        self.assertIn(b"kaiming_normal", first)

    def test_initialization_audit_renderer_writes_stable_svg(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_kaiming_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    initialization_gain=1.0,
                ),
                training_config=TrainingConfig(
                    steps=2,
                    batch_size=4,
                    learning_rate=0.01,
                    seed=26,
                ),
            )
            output = Path(directory) / "plots" / "initialization.svg"
            render_initialization_audit(output, result)
            first = output.read_bytes()
            render_initialization_audit(output, result)
            second = output.read_bytes()
        self.assertEqual(first, second)
        self.assertIn(b"Linear-weight initialization audit", first)
        self.assertIn(b"expected", first)
        self.assertIn(b"observed", first)


if __name__ == "__main__":
    unittest.main()
