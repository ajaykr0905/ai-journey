from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from ai_journey.context_mlp import build_context_dataset
from ai_journey.wavenet import (
    WaveNetBatchCursor,
    WaveNetConfig,
    WaveNetDataset,
    WaveNetTrainingConfig,
    build_wavenet_dataset_split,
    initialize_wavenet,
)
from ai_journey.wavenet_rebuild import (
    RebuiltWaveNet,
    audit_rebuild_finite_difference,
    audit_rebuild_forward,
    audit_rebuild_gradients,
    compile_rebuild_plan,
    evaluate_rebuild,
    fit_rebuild,
    initialize_rebuilt_wavenet,
    load_rebuild_checkpoint,
    load_reference_parameters,
    rebuild_model_fingerprint,
    run_rebuild_overfit_probe,
    sample_rebuild,
    save_rebuild_checkpoint,
    trace_rebuild_shapes,
    train_rebuild_steps,
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

    def test_bounded_training_records_finite_step_metrics(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            group_factors=(2, 2),
        )
        training = WaveNetTrainingConfig(steps=4, batch_size=3, seed=33)
        model = initialize_rebuilt_wavenet(config, seed=training.seed)
        cursor = WaveNetBatchCursor(
            dataset, batch_size=training.batch_size, seed=training.seed
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
        before = rebuild_model_fingerprint(model)

        trace = train_rebuild_steps(
            model, cursor, optimizer, training, start_step=5, step_count=2
        )

        self.assertEqual([step.step for step in trace], [5, 6])
        self.assertTrue(all(math.isfinite(step.loss) for step in trace))
        self.assertTrue(all(math.isfinite(step.gradient_norm) for step in trace))
        self.assertNotEqual(rebuild_model_fingerprint(model), before)

    def test_bounded_training_rejects_empty_intervals(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            group_factors=(2, 2),
        )
        model = initialize_rebuilt_wavenet(config)
        cursor = WaveNetBatchCursor(dataset, batch_size=3)
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "step_count must be positive"):
            train_rebuild_steps(
                model,
                cursor,
                optimizer,
                WaveNetTrainingConfig(),
                step_count=0,
            )

    def test_full_rebuild_training_is_deterministic(self) -> None:
        words = ("ajay", "maya", "arun", "diya", "neel")
        vocab_size = len({".", *"".join(words)})
        config = WaveNetConfig(
            vocab_size=vocab_size,
            context_size=4,
            embedding_dim=4,
            hidden_dim=8,
            group_factors=(2, 2),
        )
        datasets = build_wavenet_dataset_split(
            words, config=config, validation_fraction=0.4, seed=33
        )
        training = WaveNetTrainingConfig(steps=3, batch_size=4, seed=33)

        first = fit_rebuild(datasets, model_config=config, training_config=training)
        second = fit_rebuild(datasets, model_config=config, training_config=training)

        self.assertEqual(
            rebuild_model_fingerprint(first.model),
            rebuild_model_fingerprint(second.model),
        )
        self.assertEqual(first.trace, second.trace)
        self.assertEqual(first.cursor_state, second.cursor_state)
        self.assertTrue(math.isfinite(first.final_validation.nll))

    def test_full_rebuild_training_rejects_dataset_mismatch(self) -> None:
        words = ("ajay", "maya", "arun")
        vocab_size = len({".", *"".join(words)})
        source_config = WaveNetConfig(vocab_size=vocab_size)
        datasets = build_wavenet_dataset_split(
            words, config=source_config, validation_fraction=0.34
        )
        with self.assertRaisesRegex(ValueError, "context_size does not match"):
            fit_rebuild(
                datasets,
                model_config=WaveNetConfig(
                    vocab_size=vocab_size,
                    context_size=4,
                    group_factors=(2, 2),
                ),
                training_config=WaveNetTrainingConfig(steps=1),
            )

    def test_checkpoint_round_trip_restores_every_training_state(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            group_factors=(2, 2),
        )
        training = WaveNetTrainingConfig(steps=4, batch_size=3, seed=33)
        model = initialize_rebuilt_wavenet(config, seed=training.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)
        cursor = WaveNetBatchCursor(
            dataset, batch_size=training.batch_size, seed=training.seed
        )
        train_rebuild_steps(model, cursor, optimizer, training, step_count=2)
        expected_fingerprint = rebuild_model_fingerprint(model)
        expected_cursor = cursor.state_dict()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rebuild.pt"
            save_rebuild_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                dataset_fingerprint=dataset.fingerprint(),
                step=2,
            )
            self.assertTrue(path.is_file())
            with torch.no_grad():
                model.output_bias.add_(1)
            cursor.next()

            step = load_rebuild_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                dataset_fingerprint=dataset.fingerprint(),
            )

        self.assertEqual(step, 2)
        self.assertEqual(rebuild_model_fingerprint(model), expected_fingerprint)
        self.assertEqual(cursor.state_dict(), expected_cursor)

    def test_checkpoint_rejects_dataset_mismatch_without_mutation(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            group_factors=(2, 2),
        )
        training = WaveNetTrainingConfig(steps=1, batch_size=3)
        model = initialize_rebuilt_wavenet(config)
        optimizer = torch.optim.AdamW(model.parameters())
        cursor = WaveNetBatchCursor(dataset, batch_size=3)
        before = rebuild_model_fingerprint(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rebuild.pt"
            save_rebuild_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                dataset_fingerprint=dataset.fingerprint(),
                step=0,
            )
            with self.assertRaisesRegex(ValueError, "dataset fingerprint mismatch"):
                load_rebuild_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    training_config=training,
                    dataset_fingerprint="0" * 64,
                )
        self.assertEqual(rebuild_model_fingerprint(model), before)

    def test_checkpoint_resume_matches_uninterrupted_dropout_training(self) -> None:
        source = build_context_dataset(("ajay", "maya", "arun"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            embedding_dim=4,
            hidden_dim=8,
            group_factors=(2, 2),
            dropout=0.2,
        )
        training = WaveNetTrainingConfig(steps=4, batch_size=3, seed=33)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(777)
            uninterrupted = RebuiltWaveNet(config)
            uninterrupted_optimizer = torch.optim.AdamW(
                uninterrupted.parameters(), lr=training.learning_rate
            )
            uninterrupted_cursor = WaveNetBatchCursor(
                dataset, batch_size=training.batch_size, seed=training.seed
            )
            uninterrupted_trace = train_rebuild_steps(
                uninterrupted,
                uninterrupted_cursor,
                uninterrupted_optimizer,
                training,
            )

            torch.manual_seed(777)
            interrupted = RebuiltWaveNet(config)
            interrupted_optimizer = torch.optim.AdamW(
                interrupted.parameters(), lr=training.learning_rate
            )
            interrupted_cursor = WaveNetBatchCursor(
                dataset, batch_size=training.batch_size, seed=training.seed
            )
            first_trace = train_rebuild_steps(
                interrupted,
                interrupted_cursor,
                interrupted_optimizer,
                training,
                step_count=2,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rebuild.pt"
                save_rebuild_checkpoint(
                    path,
                    model=interrupted,
                    optimizer=interrupted_optimizer,
                    cursor=interrupted_cursor,
                    training_config=training,
                    dataset_fingerprint=dataset.fingerprint(),
                    step=2,
                )
                resumed = initialize_rebuilt_wavenet(config, seed=999)
                resumed_optimizer = torch.optim.AdamW(
                    resumed.parameters(), lr=training.learning_rate
                )
                resumed_cursor = WaveNetBatchCursor(
                    dataset, batch_size=training.batch_size, seed=training.seed
                )
                start = load_rebuild_checkpoint(
                    path,
                    model=resumed,
                    optimizer=resumed_optimizer,
                    cursor=resumed_cursor,
                    training_config=training,
                    dataset_fingerprint=dataset.fingerprint(),
                )
                second_trace = train_rebuild_steps(
                    resumed,
                    resumed_cursor,
                    resumed_optimizer,
                    training,
                    start_step=start,
                    step_count=2,
                )

        self.assertEqual(first_trace + second_trace, uninterrupted_trace)
        self.assertEqual(
            rebuild_model_fingerprint(resumed),
            rebuild_model_fingerprint(uninterrupted),
        )
        self.assertEqual(resumed_cursor.state_dict(), uninterrupted_cursor.state_dict())

    def test_sampling_is_bounded_repeatable_and_rng_isolated(self) -> None:
        vocabulary = (".", "a", "j", "m", "y")
        config = WaveNetConfig(
            vocab_size=len(vocabulary), context_size=4, group_factors=(2, 2)
        )
        model = initialize_rebuilt_wavenet(config, seed=33)
        model.train()
        torch.manual_seed(123)
        expected_next = torch.rand(4)
        torch.manual_seed(123)

        first = sample_rebuild(model, vocabulary, max_new_tokens=6, seed=99, top_k=3)
        actual_next = torch.rand(4)
        second = sample_rebuild(model, vocabulary, max_new_tokens=6, seed=99, top_k=3)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first.token_ids), 6)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(expected_next, actual_next))

    def test_sampling_validates_vocabulary_and_bounds(self) -> None:
        model = initialize_rebuilt_wavenet(WaveNetConfig(vocab_size=5))
        with self.assertRaisesRegex(ValueError, "vocabulary_tokens must match"):
            sample_rebuild(model, (".", "a"))
        with self.assertRaisesRegex(ValueError, "max_new_tokens must be positive"):
            sample_rebuild(model, (".", "a", "b", "c", "d"), max_new_tokens=0)

    def test_overfit_probe_demonstrates_bounded_capacity(self) -> None:
        source = build_context_dataset(("ajay", "maya", "arun"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            embedding_dim=4,
            hidden_dim=12,
            group_factors=(2, 2),
        )

        result = run_rebuild_overfit_probe(
            dataset,
            model_config=config,
            example_count=8,
            steps=40,
            minimum_improvement=0.5,
            seed=33,
        )

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.improvement, 0.5)

    def test_overfit_probe_rejects_oversized_subset(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=4,
            group_factors=(2, 2),
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            run_rebuild_overfit_probe(
                dataset,
                model_config=config,
                example_count=dataset.sample_count + 1,
            )


if __name__ == "__main__":
    unittest.main()
