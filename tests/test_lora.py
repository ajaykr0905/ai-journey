import unittest

from ai_journey.lora import LoRAConfig, LoRAError


class LoRAConfigTests(unittest.TestCase):
    def test_capacity_and_scaling_are_predeclared(self):
        self.assertEqual(LoRAConfig(rank=4, alpha=8).scaling, 2)

    def test_invalid_controls_cannot_silently_change_the_experiment(self):
        for name, values in {
            "rank": (True, 0, 257, 1.5),
            "alpha": (True, 0, -1, float("nan"), float("inf"), 1025),
            "dropout": (True, -0.1, 1, float("nan")),
            "seed": (True, -1, 2**32, 1.5),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(LoRAError):
                    LoRAConfig(**{name: value})
