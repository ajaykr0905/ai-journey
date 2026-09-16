from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.gradients import gradient_descent_x_squared


class GradientDescentTests(unittest.TestCase):
    def test_loss_decreases_deterministically(self) -> None:
        first = gradient_descent_x_squared()
        second = gradient_descent_x_squared()
        self.assertEqual(first, second)
        self.assertLess(float(first[-1]["loss"]), float(first[0]["loss"]))
        self.assertAlmostEqual(float(first[1]["x"]), 3.2)

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            gradient_descent_x_squared(learning_rate=0)
        with self.assertRaises(ValueError):
            gradient_descent_x_squared(steps=-1)


if __name__ == "__main__":
    unittest.main()
