from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.xor import train_xor


class XorTests(unittest.TestCase):
    def test_xor_loss_drops_and_labels_are_correct(self) -> None:
        result = train_xor()
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertLess(result["final_loss"], 0.1)
        np.testing.assert_array_equal(result["labels"], result["targets"])

    def test_training_is_deterministic(self) -> None:
        first = train_xor(epochs=2_000)
        second = train_xor(epochs=2_000)
        np.testing.assert_allclose(first["predictions"], second["predictions"])
        self.assertEqual(first["sampled_losses"], second["sampled_losses"])


if __name__ == "__main__":
    unittest.main()
