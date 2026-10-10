from __future__ import annotations

import unittest

import torch

from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerBlock,
    TransformerConfig,
    TransformerLabError,
)


class PaddedDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.config = TransformerConfig(
            vocab_size=11, block_size=5, embedding_dim=8, head_count=2, layer_count=2
        )
        self.model = DecoderLanguageModel(self.config).double().eval()

    def test_decoder_padded_batch_matches_separate_prefixes(self) -> None:
        tokens = torch.tensor([[1, 2, 8, 9, 10], [4, 5, 6, 7, 8]])
        lengths = torch.tensor([2, 5])
        logits, _ = self.model(tokens, lengths=lengths)
        for row, length in enumerate(lengths.tolist()):
            expected, _ = self.model(tokens[row : row + 1, :length])
            torch.testing.assert_close(logits[row : row + 1, :length], expected)
            self.assertEqual(torch.count_nonzero(logits[row, length:]).item(), 0)
        changed = tokens.clone()
        changed[0, 2:] = 0
        torch.testing.assert_close(
            logits, self.model(changed, lengths=lengths)[0], rtol=0, atol=0
        )

    def test_block_zeros_padding_despite_nonzero_residual_and_affine_biases(
        self,
    ) -> None:
        block = TransformerBlock(self.config).double().eval()
        with torch.no_grad():
            block.feed_forward_norm.bias.fill_(2)
            block.feed_forward.network[2].bias.fill_(3)
        output = block(
            torch.randn(2, 5, 8, dtype=torch.float64), lengths=torch.tensor([1, 3])
        )
        self.assertEqual(torch.count_nonzero(output[0, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[1, 3:]).item(), 0)

    def test_rejects_padded_batch_norm_instead_of_coupling_padding_statistics(
        self,
    ) -> None:
        config = TransformerConfig(
            vocab_size=11,
            block_size=5,
            embedding_dim=8,
            head_count=2,
            normalization_mode="scratch_batch_norm",
        )
        model = DecoderLanguageModel(config)
        with self.assertRaisesRegex(TransformerLabError, "layer_norm"):
            model(torch.tensor([[1, 2, 3]]), lengths=torch.tensor([2]))

    def test_rejects_empty_batches_and_invalid_lengths(self) -> None:
        for tokens in (
            torch.empty(0, 2, dtype=torch.long),
            torch.empty(2, 0, dtype=torch.long),
        ):
            with self.assertRaisesRegex(TransformerLabError, "non-empty"):
                self.model(tokens)
        with self.assertRaisesRegex(TransformerLabError, "lengths"):
            self.model(torch.ones(2, 3, dtype=torch.long), lengths=torch.tensor([2, 4]))
