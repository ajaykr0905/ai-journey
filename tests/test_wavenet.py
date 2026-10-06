from __future__ import annotations

import math
import unittest

import torch

from ai_journey.context_mlp import build_context_dataset
from ai_journey.wavenet import (
    FlattenConsecutive,
    HierarchicalLanguageModel,
    HierarchicalStage,
    WaveNetBatchCursor,
    WaveNetConfig,
    WaveNetDataset,
    WaveNetError,
    WaveNetMetrics,
    WaveNetTrainingConfig,
    WaveNetTrainingResult,
    build_wavenet_dataset_split,
    evaluate_wavenet,
    fit_wavenet,
    initialize_wavenet,
    trace_hierarchical_shapes,
    train_wavenet_steps,
)


class WaveNetConfigTests(unittest.TestCase):
    def test_config_exposes_receptive_field_and_stage_lengths(self) -> None:
        config = WaveNetConfig(vocab_size=27, group_factors=(2, 2, 2))

        self.assertEqual(config.receptive_field, 8)
        self.assertEqual(config.stage_lengths, (4, 2, 1))
        self.assertEqual(config.fingerprint(), config.fingerprint())

    def test_config_requires_exact_hierarchical_coverage(self) -> None:
        with self.assertRaisesRegex(
            WaveNetError, "context_size must equal the product"
        ):
            WaveNetConfig(vocab_size=27, context_size=7, group_factors=(2, 2))

    def test_config_rejects_invalid_numeric_controls(self) -> None:
        invalid = (
            {"vocab_size": 1},
            {"context_size": True},
            {"embedding_dim": 0},
            {"hidden_dim": 0},
            {"group_factors": (2, 1, 4)},
            {"group_factors": (2, True, 4)},
            {"dropout": math.nan},
            {"dropout": 1.0},
        )
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                WaveNetConfig(vocab_size=27, **overrides)

    def test_training_config_validates_every_optimization_control(self) -> None:
        config = WaveNetTrainingConfig()
        self.assertEqual(config.steps, 100)
        invalid = (
            {"steps": 0},
            {"batch_size": True},
            {"learning_rate": math.inf},
            {"weight_decay": -0.1},
            {"gradient_clip": 0.0},
            {"seed": False},
        )
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                WaveNetTrainingConfig(**overrides)


class WaveNetDatasetTests(unittest.TestCase):
    def test_context_dataset_conversion_is_exact_and_memory_independent(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        converted = WaveNetDataset.from_context_dataset(source)

        self.assertEqual(converted.sample_count, source.sample_count)
        self.assertEqual(converted.context_size, 4)
        self.assertEqual(converted.vocabulary_tokens, source.vocabulary.tokens)
        self.assertTrue(
            torch.equal(converted.contexts, torch.from_numpy(source.contexts))
        )
        original = int(converted.contexts[0, 0])
        source.contexts[0, 0] = (original + 1) % converted.vocab_size
        self.assertEqual(int(converted.contexts[0, 0]), original)

    def test_dataset_fingerprint_changes_with_targets(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        changed_targets = dataset.targets.clone()
        changed_targets[0] = (int(changed_targets[0]) + 1) % dataset.vocab_size
        changed = WaveNetDataset(
            dataset.vocabulary_tokens, dataset.contexts.clone(), changed_targets
        )

        self.assertNotEqual(dataset.fingerprint(), changed.fingerprint())

    def test_dataset_rejects_shape_dtype_and_vocabulary_errors(self) -> None:
        valid_contexts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        valid_targets = torch.tensor([1, 0], dtype=torch.long)
        invalid = (
            ((".",), valid_contexts, valid_targets),
            ((".", "a"), valid_contexts.float(), valid_targets),
            ((".", "a"), valid_contexts, valid_targets.reshape(1, 2)),
            ((".", "a"), valid_contexts[:1], valid_targets),
            ((".", "a"), torch.tensor([[0, 2]]), torch.tensor([1])),
        )
        for values in invalid:
            with (
                self.subTest(values=values),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                WaveNetDataset(*values)

    def test_record_split_is_deterministic_and_configuration_bound(self) -> None:
        words = ("ajay", "maya", "arun", "diya", "neel")
        vocab_size = len({".", *"".join(words)})
        config = WaveNetConfig(
            vocab_size=vocab_size, context_size=4, group_factors=(2, 2)
        )

        first = build_wavenet_dataset_split(
            words, config=config, validation_fraction=0.4, seed=32
        )
        second = build_wavenet_dataset_split(
            words, config=config, validation_fraction=0.4, seed=32
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.train.context_size, config.context_size)
        self.assertEqual(first.validation.vocab_size, config.vocab_size)

    def test_record_split_rejects_wrong_config_vocabulary(self) -> None:
        config = WaveNetConfig(vocab_size=99, context_size=4, group_factors=(2, 2))
        with self.assertRaisesRegex(WaveNetError, "vocab_size does not match"):
            build_wavenet_dataset_split(("ajay", "maya"), config=config)

    def test_batch_cursor_covers_each_sample_once_per_epoch(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        cursor = WaveNetBatchCursor(dataset, batch_size=3, seed=32)
        seen: list[tuple[int, ...]] = []

        while cursor.epoch == 0 and cursor.offset < dataset.sample_count:
            contexts, _ = cursor.next()
            seen.extend(tuple(int(token) for token in row) for row in contexts)

        expected = [tuple(int(token) for token in row) for row in dataset.contexts]
        self.assertCountEqual(seen, expected)
        self.assertEqual(len(seen), dataset.sample_count)

    def test_batch_cursor_resumes_at_the_exact_next_batch(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        first = WaveNetBatchCursor(dataset, batch_size=3, seed=32)
        first.next()
        state = first.state_dict()
        expected = first.next()

        resumed = WaveNetBatchCursor(dataset, batch_size=3, seed=32)
        resumed.load_state_dict(state)
        actual = resumed.next()

        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_batch_cursor_rejects_invalid_state(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        cursor = WaveNetBatchCursor(
            WaveNetDataset.from_context_dataset(source), batch_size=3
        )
        for state in (
            {"epoch": 0},
            {"epoch": -1, "offset": 0},
            {"epoch": 0, "offset": True},
            {"epoch": 0, "offset": 10_000},
        ):
            with (
                self.subTest(state=state),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                cursor.load_state_dict(state)


class FlattenConsecutiveTests(unittest.TestCase):
    def test_module_preserves_adjacent_token_order_and_gradients(self) -> None:
        inputs = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
        inputs.requires_grad_(True)
        module = FlattenConsecutive(2)

        output = module(inputs)

        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(output[0, 0].tolist(), [0, 1, 2, 3, 4, 5])
        output.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_module_handles_noncontiguous_inputs(self) -> None:
        inputs = torch.arange(2 * 3 * 4).reshape(2, 3, 4).transpose(1, 2)
        output = FlattenConsecutive(2)(inputs)

        self.assertEqual(output.shape, (2, 2, 6))
        self.assertEqual(output[0, 0].tolist(), [0, 4, 8, 1, 5, 9])

    def test_module_rejects_partial_groups_and_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(WaveNetError, "divisible"):
            FlattenConsecutive(2)(torch.zeros(3, 5, 4))
        with self.assertRaises(TypeError):
            FlattenConsecutive(2)(torch.zeros(3, 4))
        with self.assertRaises((TypeError, WaveNetError)):
            FlattenConsecutive(True)


class HierarchicalStageTests(unittest.TestCase):
    def test_stage_registers_a_complete_module_pipeline(self) -> None:
        stage = HierarchicalStage(3, 5, factor=2)
        inputs = torch.randn(4, 6, 3, requires_grad=True)

        output = stage(inputs)

        self.assertEqual(output.shape, (4, 3, 5))
        self.assertEqual(
            [type(module) for module in stage.network],
            [
                FlattenConsecutive,
                torch.nn.Linear,
                torch.nn.LayerNorm,
                torch.nn.Tanh,
                torch.nn.Dropout,
            ],
        )
        output.square().sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in stage.parameters())
        )

    def test_stage_rejects_wrong_feature_width(self) -> None:
        stage = HierarchicalStage(3, 5, factor=2)
        with self.assertRaisesRegex(WaveNetError, "feature width"):
            stage(torch.zeros(2, 4, 2))

    def test_stage_validates_dimensions_and_dropout(self) -> None:
        invalid = (
            {"input_dim": 0, "output_dim": 3, "factor": 2},
            {"input_dim": 3, "output_dim": True, "factor": 2},
            {"input_dim": 3, "output_dim": 4, "factor": 2, "dropout": 1.0},
        )
        for values in invalid:
            with (
                self.subTest(values=values),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                HierarchicalStage(**values)


class HierarchicalLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WaveNetConfig(
            vocab_size=11,
            context_size=8,
            embedding_dim=4,
            hidden_dim=12,
            group_factors=(2, 2, 2),
        )

    def test_model_reduces_full_context_and_computes_loss(self) -> None:
        model = HierarchicalLanguageModel(self.config)
        inputs = torch.randint(0, self.config.vocab_size, (5, 8))
        targets = torch.randint(0, self.config.vocab_size, (5,))

        logits, loss = model(inputs, targets)

        self.assertEqual(logits.shape, (5, self.config.vocab_size))
        self.assertEqual(len(model.stages), 3)
        self.assertGreater(model.parameter_count, 0)
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_model_parameters_include_every_registered_stage(self) -> None:
        model = HierarchicalLanguageModel(self.config)
        names = tuple(name for name, _ in model.named_parameters())

        for index in range(len(self.config.group_factors)):
            self.assertIn(f"stages.{index}.network.1.weight", names)
            self.assertIn(f"stages.{index}.network.2.weight", names)

    def test_model_rejects_invalid_contexts_and_targets(self) -> None:
        model = HierarchicalLanguageModel(self.config)
        invalid_inputs = (
            torch.zeros(2, 7, dtype=torch.long),
            torch.zeros(2, 8),
            torch.full((2, 8), self.config.vocab_size, dtype=torch.long),
        )
        for inputs in invalid_inputs:
            with (
                self.subTest(shape=inputs.shape, dtype=inputs.dtype),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                model(inputs)

        with self.assertRaises(TypeError):
            model(
                torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long)
            )

    def test_shape_trace_matches_the_registered_hierarchy(self) -> None:
        model = HierarchicalLanguageModel(self.config)
        model.train()
        model.stages[1].eval()
        original_modes = tuple(module.training for module in model.modules())

        trace = trace_hierarchical_shapes(model, batch_size=3)

        self.assertEqual(
            [step.name for step in trace],
            ["embedding", "stage_1", "stage_2", "stage_3", "output"],
        )
        self.assertEqual(
            [step.output_shape for step in trace],
            [(3, 8, 4), (3, 4, 12), (3, 2, 12), (3, 1, 12), (3, 11)],
        )
        self.assertEqual(
            tuple(module.training for module in model.modules()), original_modes
        )
        self.assertEqual(
            sum(step.parameter_count for step in trace), model.parameter_count
        )

    def test_shape_trace_validates_batch_size(self) -> None:
        model = HierarchicalLanguageModel(self.config)
        for batch_size in (True, 0):
            with (
                self.subTest(batch_size=batch_size),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                trace_hierarchical_shapes(model, batch_size=batch_size)

    def test_seeded_initialization_is_repeatable_and_isolated(self) -> None:
        torch.manual_seed(901)
        caller_state = torch.random.get_rng_state().clone()

        first = initialize_wavenet(self.config, seed=32)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), caller_state))
        second = initialize_wavenet(self.config, seed=32)
        different = initialize_wavenet(self.config, seed=33)

        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))
        self.assertTrue(
            any(
                not torch.equal(first_parameter, different_parameter)
                for first_parameter, different_parameter in zip(
                    first.parameters(), different.parameters(), strict=True
                )
            )
        )

    def test_initializer_rejects_boolean_seed(self) -> None:
        with self.assertRaises(TypeError):
            initialize_wavenet(self.config, seed=True)

    def test_evaluation_covers_every_sample_and_restores_modes(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=8)
        dataset = WaveNetDataset.from_context_dataset(source)
        config = WaveNetConfig(
            vocab_size=dataset.vocab_size,
            context_size=8,
            embedding_dim=4,
            hidden_dim=12,
            group_factors=(2, 2, 2),
        )
        model = initialize_wavenet(config, seed=32)
        model.train()
        model.stages[1].eval()
        original_modes = tuple(module.training for module in model.modules())

        metrics = evaluate_wavenet(model, dataset, batch_size=3)
        with torch.no_grad():
            _, direct_loss = model(dataset.contexts, dataset.targets)

        self.assertIsInstance(metrics, WaveNetMetrics)
        self.assertEqual(metrics.sample_count, dataset.sample_count)
        assert direct_loss is not None
        self.assertAlmostEqual(metrics.nll, float(direct_loss), places=6)
        self.assertAlmostEqual(metrics.perplexity, math.exp(metrics.nll))
        self.assertEqual(
            tuple(module.training for module in model.modules()), original_modes
        )

    def test_evaluation_rejects_dataset_contract_mismatch(self) -> None:
        source = build_context_dataset(("ajay", "maya"), block_size=4)
        dataset = WaveNetDataset.from_context_dataset(source)
        model = initialize_wavenet(self.config)
        with self.assertRaisesRegex(WaveNetError, "context_size"):
            evaluate_wavenet(model, dataset)


class WaveNetTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.words = ("ajay", "maya", "arun", "diya", "neel", "riya")
        self.config = WaveNetConfig(
            vocab_size=len({".", *"".join(self.words)}),
            context_size=4,
            embedding_dim=4,
            hidden_dim=16,
            group_factors=(2, 2),
            dropout=0.1,
        )
        self.datasets = build_wavenet_dataset_split(
            self.words, config=self.config, validation_fraction=0.33, seed=7
        )
        self.training = WaveNetTrainingConfig(
            steps=30,
            batch_size=8,
            learning_rate=0.03,
            gradient_clip=1.0,
            seed=32,
        )

    def test_training_is_repeatable_reduces_loss_and_preserves_rng(self) -> None:
        torch.manual_seed(404)
        caller_state = torch.random.get_rng_state().clone()

        first = fit_wavenet(
            self.datasets,
            model_config=self.config,
            training_config=self.training,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), caller_state))
        second = fit_wavenet(
            self.datasets,
            model_config=self.config,
            training_config=self.training,
        )

        self.assertIsInstance(first, WaveNetTrainingResult)
        self.assertLess(first.final_train.nll, first.initial_train.nll)
        self.assertEqual(first.trace, second.trace)
        self.assertEqual(len(first.trace), self.training.steps)
        self.assertTrue(
            all(
                step.gradient_norm >= 0 and math.isfinite(step.loss)
                for step in first.trace
            )
        )
        for name, first_value in first.model.state_dict().items():
            self.assertTrue(torch.equal(first_value, second.model.state_dict()[name]))

    def test_training_rejects_dataset_model_mismatch(self) -> None:
        wrong_config = WaveNetConfig(
            vocab_size=self.config.vocab_size,
            context_size=8,
            group_factors=(2, 2, 2),
        )
        with self.assertRaisesRegex(WaveNetError, "context_size"):
            fit_wavenet(
                self.datasets,
                model_config=wrong_config,
                training_config=self.training,
            )

    def test_bounded_training_continues_step_and_batch_state(self) -> None:
        config = WaveNetTrainingConfig(
            steps=4,
            batch_size=5,
            learning_rate=0.02,
            gradient_clip=1.0,
            seed=19,
        )
        model = initialize_wavenet(self.config, seed=config.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        cursor = WaveNetBatchCursor(
            self.datasets.train, batch_size=config.batch_size, seed=config.seed
        )

        first = train_wavenet_steps(
            model, cursor, optimizer, config, start_step=0, step_count=2
        )
        second = train_wavenet_steps(
            model, cursor, optimizer, config, start_step=2, step_count=2
        )

        self.assertEqual([step.step for step in (*first, *second)], [0, 1, 2, 3])
        self.assertGreater(cursor.state_dict()["offset"], 0)

    def test_bounded_training_rejects_invalid_ranges(self) -> None:
        model = initialize_wavenet(self.config)
        optimizer = torch.optim.AdamW(model.parameters())
        cursor = WaveNetBatchCursor(self.datasets.train, batch_size=4)
        for options in ({"start_step": -1}, {"step_count": 0}):
            with (
                self.subTest(options=options),
                self.assertRaises((TypeError, WaveNetError)),
            ):
                train_wavenet_steps(model, cursor, optimizer, self.training, **options)


if __name__ == "__main__":
    unittest.main()
