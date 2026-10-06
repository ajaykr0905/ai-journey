from __future__ import annotations

import math
import unittest

import torch

from ai_journey.context_mlp import build_context_dataset
from ai_journey.wavenet import (
    FlattenConsecutive,
    WaveNetConfig,
    WaveNetDataset,
    WaveNetError,
    build_wavenet_dataset_split,
)


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

    def test_record_split_is_deterministic_and_configuration_bound(self) -> None:
        words = ("ajay", "maya", "arun", "diya", "neel")
        vocab_size = len({".", *"".join(words)})
        config = WaveNetConfig(
            vocab_size=vocab_size, context_size=4, group_factors=(2, 2)
        )

        first = build_wavenet_dataset_split(
            words, config=config, validation_fraction=0.4, seed=32
        )
        second = build_wavenet_dataset_split(
            words, config=config, validation_fraction=0.4, seed=32
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.train.context_size, config.context_size)
        self.assertEqual(first.validation.vocab_size, config.vocab_size)

    def test_record_split_rejects_wrong_config_vocabulary(self) -> None:
        config = WaveNetConfig(vocab_size=99, context_size=4, group_factors=(2, 2))
        with self.assertRaisesRegex(WaveNetError, "vocab_size does not match"):
            build_wavenet_dataset_split(("ajay", "maya"), config=config)


class FlattenConsecutiveTests(unittest.TestCase):
    def test_module_preserves_adjacent_token_order_and_gradients(self) -> None:
        inputs = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
        inputs.requires_grad_(True)
        module = FlattenConsecutive(2)

        output = module(inputs)

        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(output[0, 0].tolist(), [0, 1, 2, 3, 4, 5])
        output.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_module_handles_noncontiguous_inputs(self) -> None:
        inputs = torch.arange(2 * 3 * 4).reshape(2, 3, 4).transpose(1, 2)
        output = FlattenConsecutive(2)(inputs)

        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(output[0, 0].tolist(), [0, 4, 8, 1, 5, 9])

    def test_module_rejects_partial_groups_and_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(WaveNetError, "divisible"):
            FlattenConsecutive(2)(torch.zeros(3, 5, 4))
        with self.assertRaises(TypeError):
            FlattenConsecutive(2)(torch.zeros(3, 4))
        with self.assertRaises((TypeError, WaveNetError)):
            FlattenConsecutive(True)


if __name__ == "__main__":
    unittest.main()
