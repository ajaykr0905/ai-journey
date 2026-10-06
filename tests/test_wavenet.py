from __future__ import annotations

import math
import unittest

import torch

from ai_journey.context_mlp import build_context_dataset
from ai_journey.wavenet import WaveNetConfig, WaveNetDataset, WaveNetError


class WaveNetConfigTests(unittest.TestCase):
    def test_config_exposes_receptive_field_and_stage_lengths(self) -> None:
        config = WaveNetConfig(vocab_size=27, group_factors=(2, 2, 2))

        self.assertEqual(config.receptive_field, 8)
        self.assertEqual(config.stage_lengths, (4, 2, 1))
        self.assertEqual(config.fingerprint(), config.fingerprint())

    def test_config_requires_exact_hierarchical_coverage(self) -> None:
        with self.assertRaisesRegex(
            WaveNetError, "context_size must equal the product"
        ):
            WaveNetConfig(vocab_size=27, context_size=7, group_factors=(2, 2))

    def test_config_rejects_invalid_numeric_controls(self) -> None:
        invalid = (
            {"vocab_size": 1},
            {"context_size": True},
            {"embedding_dim": 0},
            {"hidden_dim": 0},
            {"group_factors": (2, 1, 4)},
            {"group_factors": (2, True, 4)},
            {"dropout": math.nan},
            {"dropout": 1.0},
        )
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                WaveNetConfig(vocab_size=27, **overrides)


class WaveNetDatasetTests(unittest.TestCase):
    def test_context_dataset_conversion_is_exact_and_memory_independent(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        converted = WaveNetDataset.from_context_dataset(source)

        self.assertEqual(converted.sample_count, source.sample_count)
        self.assertEqual(converted.context_size, 4)
        self.assertEqual(converted.vocabulary_tokens, source.vocabulary.tokens)
        self.assertTrue(
            torch.equal(converted.contexts, torch.from_numpy(source.contexts))
        )
        original = int(converted.contexts[0, 0])
        source.contexts[0, 0] = (original + 1) % converted.vocab_size
        self.assertEqual(int(converted.contexts[0, 0]), original)

    def test_dataset_fingerprint_changes_with_targets(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        changed_targets = dataset.targets.clone()
        changed_targets[0] = (int(changed_targets[0]) + 1) % dataset.vocab_size
        changed = WaveNetDataset(
            dataset.vocabulary_tokens, dataset.contexts.clone(), changed_targets
        )

        self.assertNotEqual(dataset.fingerprint(), changed.fingerprint())

    def test_dataset_rejects_shape_dtype_and_vocabulary_errors(self) -> None:
        valid_contexts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        valid_targets = torch.tensor([1, 0], dtype=torch.long)
        invalid = (
            ((".",), valid_contexts, valid_targets),
            ((".", "a"), valid_contexts.float(), valid_targets),
            ((".", "a"), valid_contexts, valid_targets.reshape(1, 2)),
            ((".", "a"), valid_contexts[:1], valid_targets),
            ((".", "a"), torch.tensor([[0, 2]]), torch.tensor([1])),
        )
        for values in invalid:
            with (
                self.subTest(values=values),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                WaveNetDataset(*values)


if __name__ == "__main__":
    unittest.main()
