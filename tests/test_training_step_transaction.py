"""Retrying a failed CPU training step must not skip data or keep partial updates."""

import copy
import unittest
from unittest.mock import patch

import torch

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


class StepTransactionTests(unittest.TestCase):
    def initialize(self):
        seed_everything(40)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                dropout=0.2,
            )
        )
        config = TrainingConfig(steps=2, batch_size=2)
        return (
            model,
            BatchCursor(torch.arange(24) % 5, block_size=4, batch_size=2),
            build_optimizer(model, config),
            config,
        )

    def test_partial_optimizer_failure_rolls_back_and_retry_matches_reference(self):
        reference, cursor, optimizer, config = self.initialize()
        expected = train_steps(reference, cursor, optimizer, config)
        model, cursor, optimizer, config = self.initialize()
        first = train_steps(model, cursor, optimizer, config, step_count=1)
        fingerprint = model_fingerprint(model)
        cursor_before = cursor.state_dict()
        rng_before = torch.get_rng_state().clone()
        state_before = copy.deepcopy(optimizer.state_dict())
        actual_update = optimizer.step

        def fail_after_update(*args, **kwargs):
            actual_update(*args, **kwargs)
            raise RuntimeError("injected partial optimizer failure")

        with patch.object(
            optimizer, "step", side_effect=fail_after_update
        ), self.assertRaises(RuntimeError):
            train_steps(model, cursor, optimizer, config, start_step=1, step_count=1)
        self.assertEqual(model_fingerprint(model), fingerprint)
        self.assertEqual(cursor.state_dict(), cursor_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        for key, saved in state_before["state"].items():
            for field, value in saved.items():
                self.assertTrue(
                    torch.equal(value, optimizer.state_dict()["state"][key][field])
                )
        retry = train_steps(
            model, cursor, optimizer, config, start_step=1, step_count=1
        )
        self.assertEqual(first + retry, expected)
        self.assertEqual(model_fingerprint(model), model_fingerprint(reference))

    def test_nonfinite_loss_preserves_cursor_gradients_and_caller_modes(self):
        model, cursor, optimizer, config = self.initialize()
        model.eval()
        first_parameter = next(model.parameters())
        first_parameter.grad = torch.ones_like(first_parameter)
        gradients = first_parameter.grad.clone()
        loss = torch.tensor(float("nan"), requires_grad=True)
        with patch.object(
            model, "forward", return_value=(None, loss)
        ), self.assertRaises(ValueError):
            train_steps(model, cursor, optimizer, config, step_count=1)
        self.assertEqual(cursor.state_dict(), {"epoch": 0, "offset": 0})
        self.assertTrue(torch.equal(first_parameter.grad, gradients))
        self.assertFalse(model.training)
