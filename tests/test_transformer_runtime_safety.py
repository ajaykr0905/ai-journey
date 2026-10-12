"""Rejected training inputs must not corrupt reproducible state."""

import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    seed_everything,
    train_steps,
)


class RuntimeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )
        self.config = TrainingConfig(steps=2, batch_size=2)
        self.cursor = BatchCursor(torch.arange(30) % 5, block_size=4, batch_size=2)
        self.optimizer = build_optimizer(self.model, self.config)

    def test_invalid_seed_is_rejected_without_changing_any_rng(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seed_everything(seed)
            self.assertEqual(python_state, random.getstate())
            self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            with self.assertRaises(ValueError):
                TrainingConfig(seed=seed)

    def test_nonfinite_training_never_reaches_the_optimizer(self):
        before = copy.deepcopy(self.model.state_dict())
        with mock.patch.object(self.optimizer, "step") as update:
            loss = torch.tensor(float("nan"), requires_grad=True)
            with mock.patch.object(self.model, "forward", return_value=(None, loss)):
                with self.assertRaisesRegex(ValueError, "finite"):
                    train_steps(
                        self.model,
                        self.cursor,
                        self.optimizer,
                        self.config,
                        step_count=1,
                    )
            update.assert_not_called()
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[name]))
        parameter = next(self.model.parameters())
        hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaisesRegex((ValueError, RuntimeError), "finite"):
                train_steps(
                    self.model, self.cursor, self.optimizer, self.config, step_count=1
                )
        finally:
            hook.remove()
        self.assertFalse(self.optimizer.state)
