from __future__ import annotations

import math
import unittest

import torch

from ai_journey.context_mlp import build_context_dataset
from ai_journey.wavenet import WaveNetConfig, WaveNetDataset, initialize_wavenet
from ai_journey.wavenet_rebuild import (
    RebuiltWaveNet,
    audit_rebuild_finite_difference,
    audit_rebuild_forward,
    audit_rebuild_gradients,
    compile_rebuild_plan,
    evaluate_rebuild,
    initialize_rebuilt_wavenet,
    load_reference_parameters,
    rebuild_model_fingerprint,
    trace_rebuild_shapes,
)


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

    def test_seeded_initialization_is_repeatable_and_rng_isolated(self) -> None:
        config = WaveNetConfig(vocab_size=7, context_size=4, group_factors=(2, 2))
        torch.manual_seed(123)
        expected_next = torch.rand(4)
        torch.manual_seed(123)

        first = initialize_rebuilt_wavenet(config, seed=33)
        actual_next = torch.rand(4)
        second = initialize_rebuilt_wavenet(config, seed=33)
        changed = initialize_rebuilt_wavenet(config, seed=34)

        self.assertTrue(torch.equal(expected_next, actual_next))
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))
        self.assertFalse(torch.equal(first.embedding_weight, changed.embedding_weight))

    def test_seeded_initialization_validates_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "config must be WaveNetConfig"):
            initialize_rebuilt_wavenet(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "seed must be an integer"):
            initialize_rebuilt_wavenet(
                WaveNetConfig(vocab_size=7),
                seed=True,  # type: ignore[arg-type]
            )

    def test_shape_trace_records_every_primitive_boundary(self) -> None:
        model = initialize_rebuilt_wavenet(
            WaveNetConfig(
                vocab_size=7,
                context_size=4,
                embedding_dim=3,
                hidden_dim=5,
                group_factors=(2, 2),
            )
        )
        model.train()

        trace = trace_rebuild_shapes(model, batch_size=3)

        self.assertTrue(model.training)
        self.assertEqual(
            [(step.name, step.input_shape, step.output_shape) for step in trace],
            [
                ("embedding", (3, 4), (3, 4, 3)),
                ("stage_1", (3, 4, 3), (3, 2, 5)),
                ("stage_2", (3, 2, 5), (3, 1, 5)),
                ("output", (3, 5), (3, 7)),
            ],
        )
        self.assertEqual(
            sum(step.parameter_count for step in trace), model.parameter_count
        )

    def test_shape_trace_validates_model_and_batch_size(self) -> None:
        with self.assertRaisesRegex(TypeError, "model must be RebuiltWaveNet"):
            trace_rebuild_shapes(object())  # type: ignore[arg-type]
        model = initialize_rebuilt_wavenet(WaveNetConfig(vocab_size=7))
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            trace_rebuild_shapes(model, batch_size=0)

    def test_reference_parameters_map_to_every_rebuild_tensor(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )
        reference = initialize_wavenet(config, seed=17)
        rebuilt = initialize_rebuilt_wavenet(config, seed=33)

        load_reference_parameters(rebuilt, reference)

        self.assertTrue(
            torch.equal(rebuilt.embedding_weight, reference.embedding.weight)
        )
        for index, stage in enumerate(reference.stages):
            self.assertTrue(
                torch.equal(rebuilt.stage_weights[index], stage.network[1].weight)
            )
            self.assertTrue(
                torch.equal(rebuilt.stage_scales[index], stage.network[2].weight)
            )
            self.assertTrue(
                torch.equal(rebuilt.stage_biases[index], stage.network[2].bias)
            )
        self.assertTrue(torch.equal(rebuilt.output_weight, reference.output.weight))
        self.assertTrue(torch.equal(rebuilt.output_bias, reference.output.bias))

    def test_reference_parameter_mapping_rejects_mismatches(self) -> None:
        config = WaveNetConfig(vocab_size=7)
        rebuilt = initialize_rebuilt_wavenet(config)
        with self.assertRaisesRegex(TypeError, "reference must"):
            load_reference_parameters(rebuilt, object())  # type: ignore[arg-type]
        mismatched = initialize_wavenet(WaveNetConfig(vocab_size=8))
        with self.assertRaisesRegex(ValueError, "configurations must match"):
            load_reference_parameters(rebuilt, mismatched)

    def test_model_fingerprint_binds_plan_names_and_parameter_values(self) -> None:
        config = WaveNetConfig(vocab_size=7)
        first = initialize_rebuilt_wavenet(config, seed=33)
        second = initialize_rebuilt_wavenet(config, seed=33)

        self.assertEqual(
            rebuild_model_fingerprint(first), rebuild_model_fingerprint(second)
        )
        with torch.no_grad():
            second.output_bias[0].add_(1)
        self.assertNotEqual(
            rebuild_model_fingerprint(first), rebuild_model_fingerprint(second)
        )

    def test_model_fingerprint_requires_rebuild_model(self) -> None:
        with self.assertRaisesRegex(TypeError, "model must be RebuiltWaveNet"):
            rebuild_model_fingerprint(object())  # type: ignore[arg-type]

    def test_forward_audit_proves_reference_equivalence(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )
        reference = initialize_wavenet(config, seed=17)
        rebuilt = initialize_rebuilt_wavenet(config, seed=33)
        load_reference_parameters(rebuilt, reference)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([4, 5])

        audit = audit_rebuild_forward(reference, rebuilt, contexts, targets)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.examples, 2)
        self.assertLessEqual(audit.max_abs_logit_error, 1e-6)
        self.assertLessEqual(audit.loss_abs_error, 1e-6)

    def test_forward_audit_detects_parameter_drift(self) -> None:
        config = WaveNetConfig(vocab_size=7)
        reference = initialize_wavenet(config, seed=17)
        rebuilt = initialize_rebuilt_wavenet(config, seed=33)
        load_reference_parameters(rebuilt, reference)
        with torch.no_grad():
            rebuilt.output_bias[0].add_(1)
        contexts = torch.zeros((2, config.context_size), dtype=torch.long)
        targets = torch.tensor([0, 1])

        audit = audit_rebuild_forward(
            reference, rebuilt, contexts, targets, tolerance=1e-8
        )

        self.assertFalse(audit.passed)
        self.assertGreater(audit.max_abs_logit_error, audit.tolerance)

    def test_gradient_audit_covers_every_rebuild_parameter(self) -> None:
        config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=5,
            group_factors=(2, 2),
        )
        reference = initialize_wavenet(config, seed=17)
        rebuilt = initialize_rebuilt_wavenet(config, seed=33)
        load_reference_parameters(rebuilt, reference)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([4, 5])

        audit = audit_rebuild_gradients(reference, rebuilt, contexts, targets)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.parameter_tensors, len(tuple(rebuilt.parameters())))
        self.assertLessEqual(audit.max_abs_error, audit.tolerance)

    def test_gradient_audit_detects_parameter_drift(self) -> None:
        config = WaveNetConfig(vocab_size=7)
        reference = initialize_wavenet(config, seed=17)
        rebuilt = initialize_rebuilt_wavenet(config, seed=33)
        load_reference_parameters(rebuilt, reference)
        with torch.no_grad():
            rebuilt.output_bias[0].add_(1)
        contexts = torch.zeros((2, config.context_size), dtype=torch.long)
        targets = torch.tensor([0, 1])

        audit = audit_rebuild_gradients(
            reference, rebuilt, contexts, targets, tolerance=1e-8
        )

        self.assertFalse(audit.passed)
        self.assertIn("output.bias", audit.mismatched_parameters)

    def test_finite_difference_audit_checks_autograd_independently(self) -> None:
        config = WaveNetConfig(vocab_size=7, context_size=4, group_factors=(2, 2))
        model = initialize_rebuilt_wavenet(config, seed=33)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        targets = torch.tensor([4, 5])
        fingerprint = rebuild_model_fingerprint(model)

        audit = audit_rebuild_finite_difference(model, contexts, targets)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.parameter, "output_bias")
        self.assertEqual(rebuild_model_fingerprint(model), fingerprint)

    def test_finite_difference_audit_validates_parameter_selection(self) -> None:
        config = WaveNetConfig(vocab_size=7)
        model = initialize_rebuilt_wavenet(config)
        contexts = torch.zeros((2, config.context_size), dtype=torch.long)
        targets = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "unknown rebuild parameter"):
            audit_rebuild_finite_difference(
                model, contexts, targets, parameter="missing"
            )
        with self.assertRaisesRegex(TypeError, "one integer per"):
            audit_rebuild_finite_difference(model, contexts, targets, index=(0, 0))

    def test_evaluation_covers_the_complete_dataset_and_preserves_mode(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        model = initialize_rebuilt_wavenet(
            WaveNetConfig(
                vocab_size=dataset.vocab_size,
                context_size=4,
                group_factors=(2, 2),
            )
        )
        model.train()

        metrics = evaluate_rebuild(model, dataset, batch_size=3)

        self.assertTrue(model.training)
        self.assertEqual(metrics.sample_count, dataset.sample_count)
        self.assertGreater(metrics.nll, 0)
        self.assertAlmostEqual(metrics.perplexity, math.exp(metrics.nll))

    def test_evaluation_rejects_dataset_configuration_mismatch(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        model = initialize_rebuilt_wavenet(
            WaveNetConfig(vocab_size=dataset.vocab_size, context_size=8)
        )
        with self.assertRaisesRegex(ValueError, "context_size does not match"):
            evaluate_rebuild(model, dataset)


if __name__ == "__main__":
    unittest.main()
