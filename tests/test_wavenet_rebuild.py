from __future__ import annotations

import unittest

from ai_journey.wavenet import WaveNetConfig
from ai_journey.wavenet_rebuild import RebuiltWaveNet, compile_rebuild_plan


class RebuildPlanTests(unittest.TestCase):
    def test_plan_compiles_every_stage_shape_and_parameter(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )

        plan = compile_rebuild_plan(config)

        self.assertEqual(
            [
                (
                    stage.input_length,
                    stage.output_length,
                    stage.factor,
                    stage.input_dim,
                    stage.output_dim,
                    stage.weight_shape,
                )
                for stage in plan.stages
            ],
            [
                (4, 2, 2, 3, 5, (5, 6)),
                (2, 1, 2, 5, 5, (5, 10)),
            ],
        )
        self.assertEqual(
            plan.parameter_count, 7 * 3 + 5 * 6 + 10 + 5 * 10 + 10 + 5 * 7 + 7
        )
        self.assertEqual(plan.config_fingerprint, config.fingerprint())
        self.assertEqual(plan.fingerprint(), compile_rebuild_plan(config).fingerprint())

    def test_plan_requires_a_valid_wavenet_config(self) -> None:
        with self.assertRaisesRegex(TypeError, "config must be WaveNetConfig"):
            compile_rebuild_plan(object())  # type: ignore[arg-type]


class RebuiltWaveNetTests(unittest.TestCase):
    def test_model_registers_every_planned_parameter_shape(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )
        model = RebuiltWaveNet(config)

        self.assertEqual(model.parameter_count, model.plan.parameter_count)
        self.assertEqual(
            model.parameter_manifest(),
            {
                "embedding_weight": (7, 3),
                "stage_weights.0": (5, 6),
                "stage_weights.1": (5, 10),
                "stage_scales.0": (5,),
                "stage_scales.1": (5,),
                "stage_biases.0": (5,),
                "stage_biases.1": (5,),
                "output_weight": (7, 5),
                "output_bias": (7,),
            },
        )


if __name__ == "__main__":
    unittest.main()
