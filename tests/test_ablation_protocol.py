from __future__ import annotations

import unittest
from dataclasses import replace

from ai_journey.ablation_protocol import AblationArm, ControlledAblation
from ai_journey.transformer_lab import (
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
)


class ControlledAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TransformerConfig(
            vocab_size=12,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
        )
        self.training = TrainingConfig(
            steps=3,
            batch_size=4,
            learning_rate=3e-3,
            seed=31,
        )

    def protocol(self, **overrides: object) -> ControlledAblation:
        arms = (
            AblationArm("baseline", self.model, self.training),
            AblationArm(
                "low-rate",
                self.model,
                replace(self.training, learning_rate=1e-3),
            ),
            AblationArm(
                "high-rate",
                self.model,
                replace(self.training, learning_rate=6e-3),
            ),
        )
        values: dict[str, object] = {
            "name": "learning-rate-sweep",
            "hypothesis": "The baseline rate minimizes validation NLL.",
            "independent_variable": "training.learning_rate",
            "primary_metric": "validation_nll",
            "expected_direction": "lower",
            "minimum_effect": 0.01,
            "baseline_label": "baseline",
            "trial_seeds": (31, 32),
            "arms": arms,
        }
        values.update(overrides)
        return ControlledAblation(**values)  # type: ignore[arg-type]

    def test_protocol_records_hypothesis_arms_and_fixed_controls(self) -> None:
        protocol = self.protocol()
        payload = protocol.to_dict()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["baseline_label"], "baseline")
        self.assertEqual(len(payload["arms"]), 3)
        self.assertNotIn("training.learning_rate", payload["fixed_controls"])
        self.assertEqual(payload["fixed_controls"]["training.batch_size"], 4)
        self.assertEqual(payload["fixed_controls"]["paired_trial_seeds"], [31, 32])
        self.assertEqual(len(protocol.fingerprint()), 64)

    def test_fingerprint_is_stable_and_sensitive_to_the_hypothesis(self) -> None:
        protocol = self.protocol()
        self.assertEqual(protocol.fingerprint(), self.protocol().fingerprint())
        self.assertNotEqual(
            protocol.fingerprint(),
            self.protocol(hypothesis="A different prediction.").fingerprint(),
        )

    def test_rejects_an_undeclared_second_variable(self) -> None:
        arms = list(self.protocol().arms)
        arms[1] = replace(
            arms[1],
            training_config=replace(
                arms[1].training_config,
                batch_size=8,
            ),
        )
        with self.assertRaisesRegex(TransformerLabError, "training.batch_size"):
            self.protocol(arms=tuple(arms))

    def test_rejects_a_declared_variable_that_does_not_change(self) -> None:
        with self.assertRaisesRegex(
            TransformerLabError, "changed training.learning_rate"
        ):
            self.protocol(independent_variable="training.batch_size")

    def test_rejects_duplicate_independent_values(self) -> None:
        arms = self.protocol().arms
        duplicate = replace(arms[2], training_config=arms[1].training_config)
        with self.assertRaisesRegex(TransformerLabError, "values must be unique"):
            self.protocol(arms=(arms[0], arms[1], duplicate))

    def test_rejects_non_experimental_identity_and_seed_controls(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "paired controls"):
            self.protocol(independent_variable="model.vocab_size")
        with self.assertRaisesRegex(TransformerLabError, "paired controls"):
            self.protocol(independent_variable="training.seed")

    def test_rejects_duplicate_labels_and_trial_seeds(self) -> None:
        arms = self.protocol().arms
        with self.assertRaisesRegex(TransformerLabError, "labels must be unique"):
            self.protocol(arms=(arms[0], replace(arms[1], label="baseline")))
        with self.assertRaisesRegex(TransformerLabError, "trial_seeds must be unique"):
            self.protocol(trial_seeds=(31, 31))

    def test_rejects_invalid_text_thresholds_and_categories(self) -> None:
        cases = (
            ({"name": "Bad Name"}, "name must use"),
            ({"hypothesis": "  "}, "must not be blank"),
            ({"primary_metric": "accuracy"}, "primary_metric"),
            ({"expected_direction": "sideways"}, "expected_direction"),
            ({"minimum_effect": float("nan")}, "finite and non-negative"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(TransformerLabError, message),
            ):
                self.protocol(**overrides)


if __name__ == "__main__":
    unittest.main()
