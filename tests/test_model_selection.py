from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_mlp import ContextMLPError
from ai_journey.model_selection import split_train_dev_test


class CorpusPartitionTests(unittest.TestCase):
    def test_three_way_split_is_deterministic_and_disjoint(self) -> None:
        words = tuple(f"name{letter}" for letter in "abcdefghij")
        first = split_train_dev_test(
            words, development_fraction=0.2, test_fraction=0.2, seed=17
        )
        second = split_train_dev_test(
            words, development_fraction=0.2, test_fraction=0.2, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.development), len(first.test)), (6, 2, 2))
        self.assertFalse(set(first.train) & set(first.development))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.development) & set(first.test))
        self.assertEqual(set(first.train + first.development + first.test), set(words))

    def test_three_way_split_rejects_invalid_allocations(self) -> None:
        with self.assertRaises(ContextMLPError):
            split_train_dev_test(("anna", "aria"))
        with self.assertRaises(ContextMLPError):
            split_train_dev_test(
                ("anna", "aria", "navi", "priya"),
                development_fraction=0.5,
                test_fraction=0.5,
            )


if __name__ == "__main__":
    unittest.main()
