from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.drills import even_squares, run_drills, running_total


class DrillTests(unittest.TestCase):
    def test_collection_drills(self) -> None:
        self.assertEqual(even_squares(range(7)), [0, 4, 16, 36])
        self.assertEqual(list(running_total([1, 2, 3, 4])), [1, 3, 6, 10])

    def test_decorator_and_typing_drill_executes(self) -> None:
        result = run_drills()
        self.assertEqual(result["typed_add"], 7)
        self.assertEqual(result["decorator_increment"], 1)


if __name__ == "__main__":
    unittest.main()
