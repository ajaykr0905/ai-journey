"""Rejected training inputs must not corrupt reproducible state."""

import random
import unittest

import numpy as np
import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    seed_everything,
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
