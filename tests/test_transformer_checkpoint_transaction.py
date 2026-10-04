"""Rejected transformer restores must leave all caller state intact."""

from __future__ import annotations

import copy
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    train_steps,
)


class CheckpointTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.original_rng = torch.get_rng_state().clone()
        self.training = TrainingConfig(steps=3, batch_size=2, seed=37)
        self.tokens = torch.arange(30) % 5

    def tearDown(self) -> None:
        torch.set_rng_state(self.original_rng)

    def objects(self, mode: str, seed: int, steps: int):
        torch.manual_seed(seed)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                dropout=0.2,
                normalization_mode=mode,
            )
        )
        optimizer = build_optimizer(model, self.training)
        cursor = BatchCursor(
            self.tokens, block_size=4, batch_size=2, seed=self.training.seed
        )
        train_steps(model, cursor, optimizer, self.training, step_count=steps)
        return model, optimizer, cursor

    def assert_tree_equal(self, actual: Any, expected: Any) -> None:
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, Mapping):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected, strict=True):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(actual, expected)

    def snapshot(self, model, optimizer, cursor):
        return copy.deepcopy(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cursor": cursor.state_dict(),
                "rng": torch.get_rng_state(),
                "modes": [module.training for module in model.modules()],
                "gradients": [parameter.grad for parameter in model.parameters()],
            }
        )

    def test_rejected_restore_preserves_every_caller_state(self) -> None:
        defects = (
            "model_state",
            "model_fingerprint",
            "optimizer_state",
            "cursor_state",
            "torch_rng_state",
            "step",
        )
        for mode in ("layer_norm", "scratch_batch_norm"):
            for defect in defects:
                with self.subTest(mode=mode, defect=defect):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "checkpoint.pt"
                        source, source_optimizer, source_cursor = self.objects(
                            mode, 41, 1
                        )
                        save_training_checkpoint(
                            path,
                            model=source,
                            optimizer=source_optimizer,
                            cursor=source_cursor,
                            training_config=self.training,
                            corpus_fingerprint="a" * 64,
                            step=1,
                        )
                        payload = torch.load(path, weights_only=True)
                        if defect == "model_state":
                            payload[defect]["lm_head.weight"] = torch.zeros(1, 1)
                        elif defect == "model_fingerprint":
                            payload[defect] = "b" * 64
                        elif defect == "optimizer_state":
                            payload[defect]["param_groups"][0]["params"] = []
                        elif defect == "cursor_state":
                            payload[defect]["offset"] = -1
                        elif defect == "torch_rng_state":
                            payload[defect] = torch.zeros(3, dtype=torch.uint8)
                        else:
                            payload[defect] = -1
                        torch.save(payload, path)
                        model, optimizer, cursor = self.objects(mode, 89, 2)
                        model.blocks[0].attention.attention_dropout.eval()
                        before = self.snapshot(model, optimizer, cursor)
                        identities = [id(parameter) for parameter in model.parameters()]
                        with self.assertRaises((ValueError, RuntimeError)):
                            load_training_checkpoint(
                                path,
                                model=model,
                                optimizer=optimizer,
                                cursor=cursor,
                                training_config=self.training,
                                corpus_fingerprint="a" * 64,
                            )
                        self.assert_tree_equal(
                            self.snapshot(model, optimizer, cursor), before
                        )
                        self.assertEqual(
                            [id(parameter) for parameter in model.parameters()],
                            identities,
                        )

    def test_valid_restore_preserves_exact_next_step(self) -> None:
        for mode in ("layer_norm", "scratch_batch_norm"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                source, source_optimizer, source_cursor = self.objects(mode, 41, 1)
                save_training_checkpoint(
                    path,
                    model=source,
                    optimizer=source_optimizer,
                    cursor=source_cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                    step=1,
                )
                train_steps(
                    source,
                    source_cursor,
                    source_optimizer,
                    self.training,
                    start_step=1,
                    step_count=1,
                )
                expected = self.snapshot(source, source_optimizer, source_cursor)
                model, optimizer, cursor = self.objects(mode, 89, 2)
                step = load_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )
                self.assertEqual(step, 1)
                train_steps(
                    model,
                    cursor,
                    optimizer,
                    self.training,
                    start_step=step,
                    step_count=1,
                )
                self.assert_tree_equal(
                    self.snapshot(model, optimizer, cursor), expected
                )


if __name__ == "__main__":
    unittest.main()
