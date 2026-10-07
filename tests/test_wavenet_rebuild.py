from __future__ import annotations

import unittest

import torch

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

    def test_forward_executes_primitive_hierarchy_and_loss(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )
        model = RebuiltWaveNet(config)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([4, 5])

        logits, loss = model(contexts, targets)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_forward_rejects_invalid_tokens_and_targets(self) -> None:
        model = RebuiltWaveNet(
            WaveNetConfig(vocab_size=7, context_size=4, group_factors=(2, 2))
        )
        valid = torch.zeros((2, 4), dtype=torch.long)
        invalid_calls = (
            lambda: model(valid.float()),
            lambda: model(valid[:, :3]),
            lambda: model(torch.full((2, 4), 7, dtype=torch.long)),
            lambda: model(valid, torch.zeros((2, 1), dtype=torch.long)),
            lambda: model(valid, torch.zeros(2)),
            lambda: model(valid, torch.full((2,), 7, dtype=torch.long)),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()


if __name__ == "__main__":
    unittest.main()
