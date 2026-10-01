from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from ai_journey.batchnorm_experiment import evaluate_mode_nll
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
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
