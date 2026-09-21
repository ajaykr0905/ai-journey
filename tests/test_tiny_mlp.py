from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.scalar_autodiff import Value
from ai_journey.tiny_mlp import (
    MLP,
    DenseLayer,
    MLPExperiment,
    MLPValidationError,
    Neuron,
    ParameterSnapshot,
    TrainingConfig,
    TrainingExample,
    binary_accuracy,
    check_parameter_gradients,
    clipping_scale,
    experiment_metrics,
    gradient_l2_norm,
    loss_components,
    parameter_count,
    prediction,
    render_architecture_mermaid,
    render_mlp_markdown,
    restore_parameters,
    run_mlp_experiment,
    sgd_step,
    snapshot_parameters,
    tiny_dataset,
    train_mlp,
    validate_dataset,
    validate_experiment,
    validate_layer_widths,
    validate_training_config,
    zero_parameter_gradients,
)


class ValidationTests(unittest.TestCase):
    def test_layer_widths_require_positive_integers(self) -> None:
        self.assertEqual(validate_layer_widths([4, 4, 1]), (4, 4, 1))
        for widths in ((), (4, 0), (4, -1)):
            with self.subTest(widths=widths), self.assertRaises(MLPValidationError):
                validate_layer_widths(widths)
        for widths in ((True,), (3.5,)):
            with self.subTest(widths=widths), self.assertRaises(TypeError):
                validate_layer_widths(widths)  # type: ignore[arg-type]

    def test_training_config_rejects_invalid_optimizer_controls(self) -> None:
        invalid = (
            TrainingConfig(steps=0),
            TrainingConfig(learning_rate=0.0),
            TrainingConfig(l2_coefficient=-1.0),
            TrainingConfig(gradient_clip_norm=0.0),
            TrainingConfig(record_every=0),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(MLPValidationError):
                validate_training_config(config)

    def test_dataset_requires_examples_and_matching_width(self) -> None:
        with self.assertRaisesRegex(MLPValidationError, "must not be empty"):
            validate_dataset((), 3)
        with self.assertRaisesRegex(MLPValidationError, "expected 3"):
            validate_dataset((TrainingExample((1.0, 2.0), 1.0),), 3)

    def test_dataset_rejects_non_finite_inputs_and_non_binary_targets(self) -> None:
        with self.assertRaises(ValueError):
            validate_dataset((TrainingExample((1.0, 2.0, float("nan")), 1.0),), 3)
        with self.assertRaisesRegex(MLPValidationError, "-1 or \\+1"):
            validate_dataset((TrainingExample((1.0, 2.0, 3.0), 0.0),), 3)

    def test_tiny_dataset_has_four_fixed_three_feature_examples(self) -> None:
        dataset = tiny_dataset()
        self.assertEqual(len(dataset), 4)
        self.assertTrue(all(len(example.inputs) == 3 for example in dataset))
        self.assertEqual(
            [example.target for example in dataset], [1.0, -1.0, -1.0, 1.0]
        )


class ModelCompositionTests(unittest.TestCase):
    def test_neuron_initialization_is_seed_deterministic(self) -> None:
        first = Neuron(3, Random(7), name="unit")
        second = Neuron(3, Random(7), name="unit")
        self.assertEqual(
            [value.data for _, value in first.named_parameters()],
            [value.data for _, value in second.named_parameters()],
        )

    def test_neuron_checks_input_width_and_value_type(self) -> None:
        neuron = Neuron(2, Random(1), name="unit")
        with self.assertRaisesRegex(MLPValidationError, "expected 2"):
            neuron((Value(1.0),))
        with self.assertRaisesRegex(TypeError, "Value objects"):
            neuron((Value(1.0), 2.0))  # type: ignore[arg-type]

    def test_linear_neuron_matches_manual_affine_value(self) -> None:
        neuron = Neuron(2, Random(1), name="unit", nonlinear=False)
        neuron.weights[0].data = 2.0
        neuron.weights[1].data = -3.0
        neuron.bias.data = 0.5
        output = neuron((Value(4.0), Value(-2.0)))
        self.assertEqual(output.data, 14.5)

    def test_dense_layer_produces_one_output_per_neuron(self) -> None:
        layer = DenseLayer(3, 4, Random(2), name="hidden")
        outputs = layer((Value(1.0), Value(2.0), Value(3.0)))
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(isinstance(output, Value) for output in outputs))

    def test_mlp_forward_is_bounded_by_tanh(self) -> None:
        model = MLP(3, (4, 4, 1), seed=9)
        output = prediction(model, (2.0, 3.0, -1.0))
        self.assertGreaterEqual(output.data, -1.0)
        self.assertLessEqual(output.data, 1.0)

    def test_prediction_requires_scalar_output_model(self) -> None:
        with self.assertRaisesRegex(MLPValidationError, "one-output"):
            prediction(MLP(3, (2,)), (1.0, 2.0, 3.0))

    def test_parameter_count_matches_named_parameter_collection(self) -> None:
        model = MLP(3, (4, 4, 1), seed=3)
        self.assertEqual(parameter_count(3, (4, 4, 1)), 41)
        self.assertEqual(len(model.named_parameters()), 41)
        self.assertEqual(len({name for name, _ in model.named_parameters()}), 41)

    def test_parameter_snapshot_round_trip_restores_all_values(self) -> None:
        model = MLP(3, (2, 1), seed=5)
        original = snapshot_parameters(model)
        for parameter in model.parameters():
            parameter.data += 10.0
        restore_parameters(model, original)
        self.assertEqual(snapshot_parameters(model), original)

    def test_restore_rejects_incomplete_or_renamed_snapshot(self) -> None:
        model = MLP(3, (2, 1), seed=5)
        state = snapshot_parameters(model)
        with self.assertRaisesRegex(MLPValidationError, "count"):
            restore_parameters(model, state[:-1])
        renamed = (ParameterSnapshot("wrong", state[0].value), *state[1:])
        with self.assertRaisesRegex(MLPValidationError, "names"):
            restore_parameters(model, renamed)


class LossAndOptimizerTests(unittest.TestCase):
    def test_loss_components_reconcile_total_data_and_penalty(self) -> None:
        model = MLP(3, (2, 1), seed=4)
        total, data_loss, penalty, outputs = loss_components(
            model, tiny_dataset(), l2_coefficient=0.01
        )
        self.assertEqual(len(outputs), 4)
        self.assertAlmostEqual(total.data, data_loss.data + penalty.data)
        self.assertGreater(penalty.data, 0.0)

    def test_zero_l2_coefficient_produces_zero_penalty(self) -> None:
        model = MLP(3, (2, 1), seed=4)
        _, _, penalty, _ = loss_components(model, tiny_dataset())
        self.assertEqual(penalty.data, 0.0)

    def test_binary_accuracy_uses_prediction_sign(self) -> None:
        examples = tiny_dataset()
        self.assertEqual(binary_accuracy((0.1, -0.2, -3.0, 4.0), examples), 1.0)
        self.assertEqual(binary_accuracy((-0.1, -0.2, -3.0, 4.0), examples), 0.75)

    def test_zero_parameter_gradients_clears_stale_values(self) -> None:
        parameters = (Value(1.0), Value(2.0))
        parameters[0].grad = 3.0
        parameters[1].grad = -4.0
        zero_parameter_gradients(parameters)
        self.assertEqual([parameter.grad for parameter in parameters], [0.0, 0.0])

    def test_gradient_l2_norm_matches_three_four_five_triangle(self) -> None:
        parameters = (Value(1.0), Value(2.0))
        parameters[0].grad = 3.0
        parameters[1].grad = 4.0
        self.assertEqual(gradient_l2_norm(parameters), 5.0)

    def test_clipping_scale_preserves_small_and_limits_large_gradients(self) -> None:
        self.assertEqual(clipping_scale(2.0, 5.0), 1.0)
        self.assertEqual(clipping_scale(10.0, 5.0), 0.5)
        self.assertEqual(clipping_scale(10.0, None), 1.0)

    def test_sgd_step_updates_every_parameter_simultaneously(self) -> None:
        parameters = (Value(1.0), Value(-2.0))
        parameters[0].grad = 4.0
        parameters[1].grad = -6.0
        sgd_step(parameters, 0.1, gradient_scale=0.5)
        self.assertAlmostEqual(parameters[0].data, 0.8)
        self.assertAlmostEqual(parameters[1].data, -1.7)

    def test_sgd_step_rejects_empty_parameters_and_bad_scale(self) -> None:
        with self.assertRaisesRegex(MLPValidationError, "must not be empty"):
            sgd_step((), 0.1)
        with self.assertRaisesRegex(MLPValidationError, "in \\(0, 1\\]"):
            sgd_step((Value(1.0),), 0.1, gradient_scale=1.1)


class GradientAndTrainingTests(unittest.TestCase):
    def test_parameter_gradient_probes_span_model_and_pass(self) -> None:
        model = MLP(3, (4, 4, 1), seed=1709)
        before = snapshot_parameters(model)
        probes = check_parameter_gradients(model, tiny_dataset(), maximum_checks=7)
        self.assertEqual(len(probes), 7)
        self.assertEqual(probes[0].name, model.named_parameters()[0][0])
        self.assertEqual(probes[-1].name, model.named_parameters()[-1][0])
        self.assertTrue(all(probe.passed for probe in probes))
        self.assertEqual(snapshot_parameters(model), before)

    def test_gradient_probe_controls_reject_invalid_values(self) -> None:
        model = MLP(3, (2, 1), seed=1)
        with self.assertRaisesRegex(MLPValidationError, "out of range"):
            check_parameter_gradients(model, tiny_dataset(), epsilon=0.0)
        with self.assertRaisesRegex(MLPValidationError, "positive"):
            check_parameter_gradients(model, tiny_dataset(), maximum_checks=0)

    def test_training_is_deterministic_and_reduces_loss(self) -> None:
        config = TrainingConfig(steps=40, record_every=10)
        first = train_mlp(
            MLP(3, config.layer_widths, seed=config.seed), tiny_dataset(), config
        )
        second = train_mlp(
            MLP(3, config.layer_widths, seed=config.seed), tiny_dataset(), config
        )
        self.assertEqual(first, second)
        self.assertLess(first.history[-1].total_loss, first.history[0].total_loss)

    def test_training_records_step_zero_intervals_and_final_step(self) -> None:
        config = TrainingConfig(steps=25, record_every=10)
        result = train_mlp(
            MLP(3, config.layer_widths, seed=config.seed), tiny_dataset(), config
        )
        self.assertEqual([point.step for point in result.history], [0, 10, 20, 25])

    def test_training_rejects_config_architecture_mismatch(self) -> None:
        with self.assertRaisesRegex(MLPValidationError, "must match"):
            train_mlp(MLP(3, (2, 1)), tiny_dataset(), TrainingConfig())

    def test_default_experiment_reaches_full_accuracy_and_large_loss_drop(self) -> None:
        experiment = run_mlp_experiment()
        metrics = experiment_metrics(experiment)
        self.assertEqual(metrics["example_count"], 4)
        self.assertEqual(metrics["parameter_count"], 41)
        self.assertGreater(metrics["loss_reduction_fraction"], 0.98)
        self.assertEqual(metrics["final_accuracy"], 1.0)
        self.assertTrue(metrics["all_gradient_probes_pass"])

    def test_experiment_validation_rejects_failed_gradient_probe(self) -> None:
        experiment = run_mlp_experiment()
        broken_probe = replace(experiment.gradient_probes[0], passed=False)
        broken = MLPExperiment(
            experiment.input_width,
            experiment.dataset,
            (broken_probe, *experiment.gradient_probes[1:]),
            experiment.training,
        )
        with self.assertRaisesRegex(MLPValidationError, "probes failed"):
            validate_experiment(broken)

    def test_experiment_validation_rejects_non_decreasing_loss(self) -> None:
        experiment = run_mlp_experiment()
        first = experiment.training.history[0]
        broken_final = replace(
            experiment.training.history[-1], total_loss=first.total_loss
        )
        broken_training = replace(
            experiment.training,
            history=(*experiment.training.history[:-1], broken_final),
        )
        with self.assertRaisesRegex(MLPValidationError, "lower"):
            validate_experiment(replace(experiment, training=broken_training))

    def test_architecture_diagram_reports_layer_sizes_and_parameter_counts(
        self,
    ) -> None:
        diagram = render_architecture_mermaid(3, (4, 4, 1))
        self.assertIn('input["input<br/>3 scalars"]', diagram)
        self.assertIn('input -->|"16 parameters"| layer0', diagram)
        self.assertIn('layer2["layer2<br/>1 scalars"]', diagram)

    def test_markdown_reports_training_probes_predictions_and_scope(self) -> None:
        report = render_mlp_markdown(run_mlp_experiment())
        self.assertIn("# Day 17 scalar MLP training report", report)
        self.assertIn("Training trace", report)
        self.assertIn("Final predictions", report)
        self.assertIn("Parameter gradient probes", report)
        self.assertIn("does not claim lecture completion", report)


if __name__ == "__main__":
    unittest.main()
