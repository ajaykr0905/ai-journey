"""Inference must preserve mixed module modes and normalization state."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    evaluate_nll,
)


class InferenceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.original_rng = torch.get_rng_state().clone()

    def tearDown(self) -> None:
        torch.set_rng_state(self.original_rng)

    def model(self, normalization: str, training: bool):
        torch.manual_seed(84)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                dropout=0.2,
                normalization_mode=normalization,
            )
        )
        model(torch.arange(12).reshape(3, 4) % 5)
        model.train(training)
        model.blocks[0].attention_norm.train(not training)
        model.blocks[0].attention.attention_dropout.train(not training)
        return model

    def snapshot(self, model):
        return (
            [module.training for module in model.modules()],
            copy.deepcopy(model.state_dict()),
        )

    def assert_snapshot_equal(self, model, expected) -> None:
        modes, state = self.snapshot(model)
        self.assertEqual(modes, expected[0])
        self.assertEqual(set(state), set(expected[1]))
        for key, value in state.items():
            self.assertTrue(torch.equal(value, expected[1][key]), key)

    def infer(self, operation: str, model):
        if operation == "generate":
            return model.generate(
                torch.tensor([[0, 1, 2, 3]]),
                new_tokens=6,
                temperature=0.7,
                top_k=3,
                generator=torch.Generator().manual_seed(9),
            )
        return evaluate_nll(model, torch.arange(20) % 5, batch_size=3)

    def test_success_restores_every_module_mode_and_buffer(self) -> None:
        for normalization in ("layer_norm", "scratch_batch_norm"):
            for training in (True, False):
                for operation in ("generate", "evaluate"):
                    with self.subTest(
                        normalization=normalization,
                        training=training,
                        operation=operation,
                    ):
                        model = self.model(normalization, training)
                        before = self.snapshot(model)
                        expected = self.infer(operation, copy.deepcopy(model).eval())
                        actual = self.infer(operation, model)
                        if isinstance(expected, torch.Tensor):
                            self.assertTrue(torch.equal(actual, expected))
                        else:
                            self.assertEqual(actual, expected)
                        self.assert_snapshot_equal(model, before)

    def test_forward_failure_restores_every_module_mode_and_buffer(self) -> None:
        for normalization in ("layer_norm", "scratch_batch_norm"):
            for training in (True, False):
                for operation in ("generate", "evaluate"):
                    with self.subTest(
                        normalization=normalization,
                        training=training,
                        operation=operation,
                    ):
                        model = self.model(normalization, training)
                        before = self.snapshot(model)
                        with patch.object(
                            model, "forward", side_effect=RuntimeError("forward failed")
                        ):
                            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                                self.infer(operation, model)
                        self.assert_snapshot_equal(model, before)


if __name__ == "__main__":
    unittest.main()
