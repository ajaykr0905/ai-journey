from __future__ import annotations

import unittest

import torch

from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    TransformerBlock,
    TransformerLabError,
)


class GenerationInputGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=7, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )

    def test_zero_token_generation_validates_and_clones_full_prompt(self) -> None:
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        generated = self.model.generate(prompt, new_tokens=0)
        torch.testing.assert_close(generated, prompt)
        self.assertNotEqual(generated.data_ptr(), prompt.data_ptr())
        invalid = prompt.clone()
        invalid[0, 0] = 7
        with self.assertRaisesRegex(TransformerLabError, "vocabulary"):
            self.model.generate(invalid, new_tokens=0)

    def test_rejects_invalid_temperature_even_without_new_tokens(self) -> None:
        for temperature in (
            True,
            False,
            0,
            -1,
            float("nan"),
            float("inf"),
            -float("inf"),
            "hot",
        ):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                TransformerLabError, "temperature"
            ):
                self.model.generate(
                    torch.tensor([[1]]), new_tokens=0, temperature=temperature
                )

    def test_rejects_empty_malformed_and_out_of_vocabulary_prompts(self) -> None:
        for prompt in (
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(0, 1, dtype=torch.long),
            torch.tensor([[-1]]),
            torch.tensor([[7]]),
        ):
            with self.subTest(prompt=prompt), self.assertRaises(TransformerLabError):
                self.model.generate(prompt, new_tokens=0)
        for prompt in (torch.tensor([[1.0]]), torch.tensor([1]), [[1]]):
            with self.assertRaises(TypeError):
                self.model.generate(prompt, new_tokens=0)

    def test_nonfinite_sampling_logits_fail_and_restore_mixed_module_modes(
        self,
    ) -> None:
        self.model.train()
        self.model.blocks[0].attention.eval()
        modes = [module.training for module in self.model.modules()]
        with torch.no_grad():
            self.model.lm_head.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(TransformerLabError, "sampling logits"):
            self.model.generate(torch.tensor([[1]]), new_tokens=1)
        self.assertEqual([module.training for module in self.model.modules()], modes)

    def test_seeded_generation_supports_long_prompts_and_preserves_inputs(self) -> None:
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        original = prompt.clone()
        first = self.model.generate(
            prompt, new_tokens=3, generator=torch.Generator().manual_seed(37)
        )
        second = self.model.generate(
            prompt, new_tokens=3, generator=torch.Generator().manual_seed(37)
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(prompt, original, rtol=0, atol=0)
        self.assertEqual(first.shape, (1, 8))


class NormalizationPlacementTests(unittest.TestCase):
    def test_block_implements_each_declared_residual_equation(self) -> None:
        for placement in ("pre", "post"):
            with self.subTest(placement=placement):
                config = TransformerConfig(
                    vocab_size=7,
                    embedding_dim=8,
                    head_count=2,
                    normalization_placement=placement,
                )
                block = TransformerBlock(config).double().eval()
                inputs = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
                actual = block(inputs)
                if placement == "pre":
                    residual = inputs + block.attention(block.attention_norm(inputs))
                    expected = residual + block.feed_forward(
                        block.feed_forward_norm(residual)
                    )
                else:
                    residual = block.attention_norm(inputs + block.attention(inputs))
                    expected = block.feed_forward_norm(
                        residual + block.feed_forward(residual)
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grad = torch.autograd.grad(
                    actual.square().sum(), inputs, retain_graph=True
                )[0]
                expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_legacy_defaults_preserve_config_fingerprint(self) -> None:
        import json
        from dataclasses import asdict
        from hashlib import sha256

        config = TransformerConfig(vocab_size=7)
        legacy = asdict(config)
        legacy.pop("normalization_placement")
        expected = sha256(
            json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(config.fingerprint(), expected)
        self.assertNotEqual(
            config.fingerprint(),
            TransformerConfig(
                vocab_size=7, normalization_placement="post"
            ).fingerprint(),
        )

    def test_post_norm_keeps_padding_zero_and_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "normalization_placement"):
            TransformerConfig(vocab_size=7, normalization_placement="sandwich")
        block = TransformerBlock(
            TransformerConfig(
                vocab_size=7,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                normalization_placement="post",
            )
        )
        output = block(torch.randn(2, 4, 8), lengths=torch.tensor([1, 3]))
        self.assertEqual(torch.count_nonzero(output[0, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[1, 3:]).item(), 0)
