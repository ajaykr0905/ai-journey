"""Independent numerical and causal oracles for trained adapter paths."""

import unittest

import torch
from torch import nn
from torch.func import functional_call

from ai_journey.lora import LoRAConfig, LoRALinear, inject_adapters
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class LoRANumericalOracleTests(unittest.TestCase):
    def test_finite_differences_verify_both_factors_and_the_input_jacobian(self):
        layer = LoRALinear(nn.Linear(3, 2).double(), LoRAConfig(rank=2, alpha=3))
        with torch.no_grad():
            layer.lora_b.copy_(torch.tensor([[0.1, -0.2], [0.3, 0.4]]))
        x = torch.tensor([[0.5, -0.3, 0.8]], dtype=torch.float64, requires_grad=True)
        a = layer.lora_a.detach().clone().requires_grad_()
        b = layer.lora_b.detach().clone().requires_grad_()

        def forward(inputs, factor_a, factor_b):
            return functional_call(
                layer, {"lora_a": factor_a, "lora_b": factor_b}, (inputs,)
            )

        self.assertTrue(
            torch.autograd.gradcheck(forward, (x, a, b), eps=1e-6, atol=1e-6)
        )

    def test_factor_gradients_match_an_independent_matrix_calculus_oracle(self):
        layer = LoRALinear(nn.Linear(3, 2).double(), LoRAConfig(rank=2, alpha=5))
        x = torch.tensor([[1, -2, 3], [0.5, 0.8, -0.1]], dtype=torch.float64)
        upstream = torch.tensor([[0.2, 0.5], [-0.3, 0.4]], dtype=torch.float64)
        with torch.no_grad():
            layer.lora_b.copy_(torch.tensor([[0.2, -0.3], [0.4, 0.1]]))
        (layer(x) * upstream).sum().backward()
        expected_a = 2.5 * (upstream @ layer.lora_b.detach()).T @ x
        expected_b = 2.5 * upstream.T @ (x @ layer.lora_a.detach().T)
        torch.testing.assert_close(
            layer.lora_a.grad, expected_a, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            layer.lora_b.grad, expected_b, rtol=1e-12, atol=1e-12
        )
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNone(layer.base.bias.grad)

    def test_trained_attention_adapters_preserve_causal_prefix_and_padding(self):
        base = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=8, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        ).double()
        model = inject_adapters(
            base,
            ("blocks.0.attention.query_key_value", "blocks.0.attention.projection"),
            LoRAConfig(),
        ).eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.fill_(0.25)
        original = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 6, 7]])
        torch.testing.assert_close(
            model(original)[0][:, :2], model(changed_future)[0][:, :2], rtol=0, atol=0
        )
        padded = model(original, lengths=torch.tensor([2]))[0][:, :2]
        unpadded = model(original[:, :2])[0]
        torch.testing.assert_close(padded, unpadded, rtol=1e-12, atol=1e-12)
