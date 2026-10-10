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

    def test_padded_loss_is_valid_token_weighted_not_batch_weighted(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        lengths = torch.tensor([2, 5])
        targets = torch.tensor([[2, 3, -100, -100, -100], [7, 8, 9, 10, 1]])
        _, loss = self.model(tokens, targets, lengths=lengths)
        losses = []
        for row, length in enumerate(lengths.tolist()):
            _, individual = self.model(
                tokens[row : row + 1, :length], targets[row : row + 1, :length]
            )
            losses.append(individual * length)
        torch.testing.assert_close(loss, sum(losses) / lengths.sum())
        changed = targets.clone()
        changed[0, 2:] = 10000
        torch.testing.assert_close(
            loss, self.model(tokens, changed, lengths=lengths)[1], rtol=0, atol=0
        )

    def test_padded_loss_gradients_match_separate_valid_prefixes(self) -> None:
        import copy

        separate = copy.deepcopy(self.model)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[2, -100, -100], [5, 6, 7]])
        _, loss = self.model(tokens, targets, lengths=torch.tensor([1, 3]))
        loss.backward()
        losses = [
            separate(tokens[0:1, :1], targets[0:1, :1])[1],
            separate(tokens[1:2], targets[1:2])[1] * 3,
        ]
        (sum(losses) / 4).backward()
        for actual, expected in zip(
            self.model.parameters(), separate.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=1e-9, atol=1e-10
            )

    def test_rejects_invalid_valid_targets_and_target_device(self) -> None:
        tokens = torch.tensor([[1, 2, 3]])
        for targets in (torch.tensor([[2, -100, 4]]), torch.tensor([[2, 11, 4]])):
            with self.assertRaisesRegex(TransformerLabError, "valid target"):
                self.model(tokens, targets, lengths=torch.tensor([2]))
        with self.assertRaisesRegex(TransformerLabError, "device"):
            self.model(tokens, torch.ones(1, 3, dtype=torch.long, device="meta"))
        with self.assertRaisesRegex(TypeError, "targets"):
            self.model(tokens, [[2, 3, 4]])
