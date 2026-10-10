from __future__ import annotations

import unittest

import torch

from ai_journey.attention_audit import per_head_reference
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


if __name__ == "__main__":
    unittest.main()
