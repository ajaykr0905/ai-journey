from __future__ import annotations

import math
import unittest

from ai_journey.wavenet import WaveNetConfig, WaveNetError


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
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, WaveNetError)
            ):
                WaveNetConfig(vocab_size=27, **overrides)


if __name__ == "__main__":
    unittest.main()
