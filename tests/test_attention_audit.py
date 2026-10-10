from __future__ import annotations

import unittest

import torch

from ai_journey.attention_audit import (
    audit_attention_causality,
    audit_attention_gradients,
    per_head_reference,
)
from ai_journey.transformer_lab import CausalSelfAttention, TransformerConfig


class AttentionReferenceTests(unittest.TestCase):
    def make_problem(self, heads=2, time=5):
        with torch.random.fork_rng():
            torch.manual_seed(37)
            module = (
                CausalSelfAttention(
                    TransformerConfig(vocab_size=7, embedding_dim=8, head_count=heads)
                )
                .double()
                .eval()
            )
            inputs = torch.randn(2, time, 8, dtype=torch.float64)
        return module, inputs

    def test_independent_heads_match_packed_attention(self):
        for heads in (1, 2, 4, 8):
            for time in (1, 5, 16):
                with self.subTest(heads=heads, time=time):
                    module, inputs = self.make_problem(heads, time)
                    reference = per_head_reference(module, inputs)
                    torch.testing.assert_close(
                        reference.output, module(inputs), atol=1e-12, rtol=1e-12
                    )
                    self.assertEqual(reference.weights.shape, (2, heads, time, time))
                    torch.testing.assert_close(
                        reference.weights.sum(-1),
                        torch.ones(2, heads, time, dtype=torch.float64),
                    )
                    self.assertEqual(
                        float(reference.weights.detach().triu(1).abs().max()), 0
                    )

    def test_reference_rejects_training_and_bad_tensors(self):
        module, inputs = self.make_problem()
        module.train()
        with self.assertRaisesRegex(ValueError, "evaluation"):
            per_head_reference(module, inputs)
        module.eval()
        for bad in (
            inputs.float(),
            inputs[:, :0],
            inputs[:, :, :7],
            inputs * float("nan"),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                per_head_reference(module, bad)

    def test_input_and_every_projection_gradient_match_independent_oracle(self):
        for heads in (1, 2, 4):
            with self.subTest(heads=heads):
                module, inputs = self.make_problem(heads)
                inputs.requires_grad_()
                inputs.grad = torch.full_like(inputs, 7)
                for parameter in module.parameters():
                    parameter.grad = torch.full_like(parameter, 9)
                state = torch.get_rng_state().clone()
                audit = audit_attention_gradients(module, inputs)
                self.assertLess(audit.maximum_error, 1e-12)
                self.assertTrue(torch.equal(inputs.grad, torch.full_like(inputs, 7)))
                self.assertTrue(
                    all(
                        torch.equal(p.grad, torch.full_like(p, 9))
                        for p in module.parameters()
                    )
                )
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertFalse(module.training)

    def test_gradient_audit_rejects_frozen_parameters(self):
        module, inputs = self.make_problem()
        module.projection.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "trainable"):
            audit_attention_gradients(module, inputs)

    def test_every_future_boundary_is_isolated_in_forward_and_backward(self):
        module, inputs = self.make_problem()
        result = audit_attention_causality(module, inputs)
        self.assertEqual(result.boundaries_checked, 4)
        self.assertEqual(result.prefix_error, 0)
        self.assertEqual(result.future_gradient, 0)
        singleton = audit_attention_causality(module, inputs[:, :1])
        self.assertEqual(singleton.boundaries_checked, 0)

    def test_causality_audit_detects_a_deliberate_future_leak(self):
        module, inputs = self.make_problem()
        original = module.forward
        module.forward = lambda x: original(x) + x.mean(dim=1, keepdim=True)
        result = audit_attention_causality(module, inputs)
        self.assertGreater(result.prefix_error, 1)
        self.assertGreater(result.future_gradient, 0)


if __name__ == "__main__":
    unittest.main()
