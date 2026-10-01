from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import ai_journey
from ai_journey.batchnorm_experiment import (
    BatchNormCriteria,
    build_batchnorm_report,
    evaluate_batchnorm_experiment,
    evaluate_mode_nll,
    render_batchnorm_diagnostics,
    run_batchnorm_experiment,
    write_batchnorm_report,
)
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    model_fingerprint,
    seed_everything,
    train_steps,
)


class PublicApiTests(unittest.TestCase):
    def test_batchnorm_capabilities_are_exported(self) -> None:
        self.assertIs(ai_journey.BatchNormCriteria, BatchNormCriteria)
        self.assertIs(ai_journey.run_batchnorm_experiment, run_batchnorm_experiment)
        self.assertEqual(ai_journey.ScratchBatchNorm.__name__, "ScratchBatchNorm")


class ModeEvaluationTests(unittest.TestCase):
    def _trained_model(self, normalization_mode: str) -> DecoderLanguageModel:
        seed_everything(27)
        config = TransformerConfig(
            vocab_size=5,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            normalization_mode=normalization_mode,
        )
        training = TrainingConfig(steps=2, batch_size=4, seed=27)
        model = DecoderLanguageModel(config)
        cursor = BatchCursor(
            torch.arange(48) % 5,
            block_size=config.block_size,
            batch_size=training.batch_size,
            seed=training.seed,
        )
        train_steps(model, cursor, build_optimizer(model, training), training)
        return model

    def test_batchnorm_train_and_eval_modes_produce_different_nll(self) -> None:
        model = self._trained_model("scratch_batch_norm")
        before = model_fingerprint(model)
        tokens = torch.arange(32) % model.config.vocab_size
        eval_nll = evaluate_mode_nll(model, tokens, training_mode=False, batch_size=3)
        train_nll = evaluate_mode_nll(model, tokens, training_mode=True, batch_size=3)
        self.assertNotEqual(eval_nll, train_nll)
        self.assertEqual(model_fingerprint(model), before)
        self.assertTrue(model.training)

    def test_layernorm_is_mode_invariant_without_dropout(self) -> None:
        model = self._trained_model("layer_norm")
        tokens = torch.arange(32) % model.config.vocab_size
        self.assertEqual(
            evaluate_mode_nll(model, tokens, training_mode=False, batch_size=3),
            evaluate_mode_nll(model, tokens, training_mode=True, batch_size=3),
        )

    def test_mode_evaluation_validates_controls(self) -> None:
        model = self._trained_model("scratch_batch_norm")
        with self.assertRaisesRegex(TypeError, "one-dimensional"):
            evaluate_mode_nll(model, torch.ones(2, 2), training_mode=False)
        with self.assertRaisesRegex(TypeError, "boolean"):
            evaluate_mode_nll(model, torch.arange(20) % 5, training_mode=1)  # type: ignore[arg-type]


class BatchNormExperimentTests(unittest.TestCase):
    def test_experiment_is_bitwise_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            model_config = TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                normalization_mode="scratch_batch_norm",
            )
            training_config = TrainingConfig(
                steps=1, batch_size=4, learning_rate=0.01, seed=27
            )
            first = run_batchnorm_experiment(
                corpus,
                model_config=model_config,
                training_config=training_config,
            )
            second = run_batchnorm_experiment(
                corpus,
                model_config=model_config,
                training_config=training_config,
            )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.initial_model_fingerprint, second.initial_model_fingerprint
        )
        self.assertEqual(first.first_batch_fingerprint, second.first_batch_fingerprint)
        self.assertEqual(first.model_fingerprint, second.model_fingerprint)

    def test_experiment_records_training_mode_trap_and_layer_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            result = run_batchnorm_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    normalization_mode="scratch_batch_norm",
                ),
                training_config=TrainingConfig(
                    steps=2, batch_size=4, learning_rate=0.01, seed=27
                ),
            )
        payload = result.to_dict()
        fingerprint = payload.pop("evidence_fingerprint")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            fingerprint,
            sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        self.assertEqual(fingerprint, result.evidence_fingerprint())
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(len(result.layer_state_fingerprints), 3)
        self.assertEqual(len(result.model_fingerprint), 64)
        self.assertEqual(len(result.initial_model_fingerprint), 64)
        self.assertEqual(len(result.first_batch_fingerprint), 64)
        self.assertEqual(result.runtime.device, "cpu")
        self.assertTrue(result.runtime.python_version)
        self.assertTrue(result.runtime.torch_version)
        self.assertTrue(result.runtime.numpy_version)
        self.assertTrue(result.runtime.machine)
        self.assertTrue(result.runtime.deterministic_algorithms)
        self.assertGreater(result.runtime.intraop_threads, 0)
        self.assertEqual(len(result.corpus_fingerprint), 64)
        self.assertNotEqual(result.mode_trap.train_minus_eval_nll, 0.0)
        self.assertGreater(result.batch_coupling.train_max_abs_delta, 0.0)
        self.assertEqual(result.batch_coupling.eval_max_abs_delta, 0.0)
        self.assertEqual(
            payload["model_config"]["normalization_mode"], "scratch_batch_norm"
        )

    def test_experiment_rejects_layernorm_and_vocabulary_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            with self.assertRaisesRegex(ValueError, "scratch_batch_norm"):
                run_batchnorm_experiment(
                    corpus,
                    model_config=TransformerConfig(vocab_size=corpus.vocab_size),
                    training_config=TrainingConfig(steps=1),
                )

    def test_experiment_restores_caller_random_state(self) -> None:
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        torch.use_deterministic_algorithms(False)
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(path, block_size=4)
            run_batchnorm_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    normalization_mode="scratch_batch_norm",
                ),
                training_config=TrainingConfig(steps=1, batch_size=4, seed=27),
            )
        actual = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_evaluation_reports_named_threshold_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(path, block_size=4)
            result = run_batchnorm_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    normalization_mode="scratch_batch_norm",
                ),
                training_config=TrainingConfig(
                    steps=2, batch_size=4, learning_rate=0.01, seed=27
                ),
            )
        loose = evaluate_batchnorm_experiment(
            result,
            BatchNormCriteria(
                min_training_loss_reduction=0.0,
                min_mode_nll_gap=0.0,
                min_train_batch_coupling=0.0,
                max_eval_batch_coupling=1.0,
            ),
        )
        self.assertTrue(loose.passed, loose.violations)
        strict = evaluate_batchnorm_experiment(
            result,
            BatchNormCriteria(
                min_training_loss_reduction=10.0,
                min_mode_nll_gap=10.0,
                min_train_batch_coupling=10.0,
                max_eval_batch_coupling=0.0,
            ),
        )
        self.assertFalse(strict.passed)
        self.assertEqual(
            strict.violations,
            (
                "training_loss_reduction",
                "mode_nll_gap",
                "train_batch_coupling",
            ),
        )

    def test_criteria_reject_invalid_thresholds(self) -> None:
        for value in (-1.0, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BatchNormCriteria(min_mode_nll_gap=value)

    def test_report_writer_is_stable_atomic_and_self_verifying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            result = run_batchnorm_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    normalization_mode="scratch_batch_norm",
                ),
                training_config=TrainingConfig(steps=2, batch_size=4, seed=27),
            )
            criteria = BatchNormCriteria(
                min_training_loss_reduction=0.0,
                min_mode_nll_gap=0.0,
                min_train_batch_coupling=0.0,
            )
            path = root / "nested" / "report.json"
            write_batchnorm_report(path, result, criteria)
            first = path.read_bytes()
            write_batchnorm_report(path, result, criteria)
            second = path.read_bytes()
            payload = json.loads(second)
            report_fingerprint = payload.pop("report_fingerprint")
            expected = sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        self.assertEqual(first, second)
        self.assertEqual(report_fingerprint, expected)
        self.assertEqual(
            payload["experiment"]["evidence_fingerprint"],
            result.evidence_fingerprint(),
        )
        self.assertTrue(payload["evaluation"]["passed"])
        self.assertFalse(path.with_name(".report.json.tmp").exists())
        expected_payload = build_batchnorm_report(result, criteria)
        expected_payload.pop("report_fingerprint")
        self.assertEqual(payload, json.loads(json.dumps(expected_payload)))

    def test_diagnostic_renderer_writes_stable_atomic_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            result = run_batchnorm_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                    normalization_mode="scratch_batch_norm",
                ),
                training_config=TrainingConfig(steps=1, batch_size=4, seed=27),
            )
            path = root / "nested" / "diagnostics.svg"
            render_batchnorm_diagnostics(path, result)
            first = path.read_bytes()
            render_batchnorm_diagnostics(path, result)
            second = path.read_bytes()
            rendered = second.decode()
            self.assertFalse(path.with_name(".diagnostics.svg.tmp").exists())
        self.assertEqual(first, second)
        self.assertIn("correct eval() NLL", rendered)
        self.assertIn("mistaken train() NLL", rendered)
        self.assertIn("train() companion delta", rendered)
        self.assertIn("CPU-only deterministic experiment", rendered)
