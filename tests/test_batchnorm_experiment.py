from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from ai_journey.batchnorm_experiment import evaluate_mode_nll, run_batchnorm_experiment
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
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(len(result.layer_state_fingerprints), 3)
        self.assertEqual(len(result.model_fingerprint), 64)
        self.assertEqual(len(result.initial_model_fingerprint), 64)
        self.assertEqual(len(result.first_batch_fingerprint), 64)
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
