from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from math import exp, tanh
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.scalar_autodiff import (  # noqa: E402
    AutodiffExperiment,
    GraphCycleError,
    Value,
    default_experiment_inputs,
    experiment_metrics,
    gradient_check,
    numerical_derivative,
    render_autodiff_markdown,
    render_mermaid,
    run_autodiff_experiment,
    snapshot_graph,
    tiny_neuron_loss,
    topological_sort,
    validate_experiment,
    zero_grad,
)


class ValueArithmeticTests(unittest.TestCase):
    def test_constructor_accepts_finite_real_scalars(self) -> None:
        self.assertEqual(Value(2).data, 2.0)
        self.assertIn("data=2", repr(Value(2.0, label="x")))

    def test_constructor_rejects_boolean_non_real_and_non_finite_data(self) -> None:
        for value in (True, "2", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Value(value)  # type: ignore[arg-type]

    def test_addition_records_value_and_unit_local_gradients(self) -> None:
        left, right = Value(2.0), Value(-3.0)
        output = left + right
        output.backward()
        self.assertEqual(output.data, -1.0)
        self.assertEqual((left.grad, right.grad), (1.0, 1.0))

    def test_multiplication_records_value_and_cross_local_gradients(self) -> None:
        left, right = Value(2.0), Value(-3.0)
        output = left * right
        output.backward()
        self.assertEqual(output.data, -6.0)
        self.assertEqual((left.grad, right.grad), (-3.0, 2.0))

    def test_reverse_operators_accept_python_scalars(self) -> None:
        value = Value(2.0)
        self.assertEqual((3.0 + value).data, 5.0)
        self.assertEqual((3.0 * value).data, 6.0)
        self.assertEqual((3.0 - value).data, 1.0)
        self.assertEqual((8.0 / value).data, 4.0)

    def test_subtraction_and_negation_propagate_signs(self) -> None:
        left, right = Value(2.0), Value(-3.0)
        output = -left - right
        output.backward()
        self.assertEqual(output.data, 1.0)
        self.assertEqual((left.grad, right.grad), (-1.0, -1.0))

    def test_power_uses_the_power_rule(self) -> None:
        value = Value(3.0)
        output = value**2
        output.backward()
        self.assertEqual(output.data, 9.0)
        self.assertEqual(value.grad, 6.0)

    def test_zero_power_has_zero_derivative_even_at_zero(self) -> None:
        value = Value(0.0)
        output = value**0
        output.backward()
        self.assertEqual((output.data, value.grad), (1.0, 0.0))

    def test_division_uses_reciprocal_chain_rule(self) -> None:
        numerator, denominator = Value(6.0), Value(3.0)
        output = numerator / denominator
        output.backward()
        self.assertAlmostEqual(output.data, 2.0)
        self.assertAlmostEqual(numerator.grad, 1.0 / 3.0)
        self.assertAlmostEqual(denominator.grad, -2.0 / 3.0)

    def test_division_by_zero_fails_explicitly(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            Value(1.0) / 0.0

    def test_exp_matches_standard_library_value_and_derivative(self) -> None:
        value = Value(0.4)
        output = value.exp()
        output.backward()
        self.assertAlmostEqual(output.data, exp(0.4))
        self.assertAlmostEqual(value.grad, exp(0.4))

    def test_tanh_matches_standard_library_value_and_derivative(self) -> None:
        value = Value(-0.7)
        output = value.tanh()
        output.backward()
        self.assertAlmostEqual(output.data, tanh(-0.7))
        self.assertAlmostEqual(value.grad, 1.0 - tanh(-0.7) ** 2)

    def test_relu_gradient_is_zero_at_and_below_zero(self) -> None:
        for point in (-2.0, 0.0):
            with self.subTest(point=point):
                value = Value(point)
                output = value.relu()
                output.backward()
                self.assertEqual((output.data, value.grad), (0.0, 0.0))

    def test_relu_gradient_is_one_above_zero(self) -> None:
        value = Value(2.0)
        output = value.relu()
        output.backward()
        self.assertEqual((output.data, value.grad), (2.0, 1.0))

    def test_shared_subexpression_accumulates_all_gradient_paths(self) -> None:
        value = Value(3.0)
        output = value * value + value
        output.backward()
        self.assertEqual(value.grad, 7.0)

    def test_backward_accepts_non_unit_seed(self) -> None:
        value = Value(4.0)
        output = value * 2.0
        output.backward(gradient=0.25)
        self.assertEqual((output.grad, value.grad), (0.25, 0.5))

    def test_backward_clears_stale_gradients_by_default(self) -> None:
        value = Value(4.0)
        output = value * 2.0
        output.backward()
        output.backward()
        self.assertEqual(value.grad, 2.0)


class GraphTests(unittest.TestCase):
    def test_topological_sort_places_parents_before_result(self) -> None:
        left, right = Value(2.0), Value(3.0)
        product = left * right
        output = product + left
        ordered = topological_sort(output)
        positions = {node: index for index, node in enumerate(ordered)}
        self.assertLess(positions[left], positions[product])
        self.assertLess(positions[right], positions[product])
        self.assertLess(positions[product], positions[output])

    def test_topological_sort_returns_shared_node_once(self) -> None:
        value = Value(2.0)
        ordered = topological_sort(value * value)
        self.assertEqual(sum(node is value for node in ordered), 1)

    def test_cycle_detection_rejects_malformed_internal_graph(self) -> None:
        value = Value(1.0)
        value._parents = (value,)
        with self.assertRaises(GraphCycleError):
            topological_sort(value)

    def test_zero_grad_clears_the_whole_reachable_graph(self) -> None:
        value = Value(2.0)
        output = (value * 3.0).tanh()
        output.backward()
        zero_grad(output)
        self.assertTrue(all(node.grad == 0.0 for node in topological_sort(output)))

    def test_snapshot_has_stable_ids_known_edges_and_gradients(self) -> None:
        left, right = Value(2.0, label="left"), Value(3.0, label="right")
        output = left * right
        output.label = "product"
        output.backward()
        graph = snapshot_graph(output)
        self.assertEqual([node.node_id for node in graph.nodes], ["n0", "n1", "n2"])
        self.assertEqual(graph.edges, (("n0", "n2"), ("n1", "n2")))
        self.assertEqual(graph.nodes[0].grad, 3.0)

    def test_mermaid_contains_labels_values_gradients_and_edges(self) -> None:
        value = Value(2.0, label="input")
        output = value + 1.0
        output.backward()
        diagram = render_mermaid(snapshot_graph(output))
        self.assertIn("flowchart LR", diagram)
        self.assertIn("input<br/>value=2.000000<br/>grad=1.000000", diagram)
        self.assertIn("n0 --> n2", diagram)


class GradientCheckTests(unittest.TestCase):
    def test_centered_difference_matches_polynomial_derivative(self) -> None:
        derivative = numerical_derivative(lambda x: x**3, 2.0)
        self.assertAlmostEqual(derivative, 12.0, places=5)

    def test_centered_difference_rejects_non_positive_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            numerical_derivative(lambda x: x, 1.0, epsilon=0.0)

    def test_gradient_check_passes_composite_expression(self) -> None:
        checks = gradient_check(
            lambda v: ((v["a"] * v["b"] + v["c"]).tanh() ** 2.0),
            {"a": 0.7, "b": -0.4, "c": 1.2},
        )
        self.assertEqual([check.name for check in checks], ["a", "b", "c"])
        self.assertTrue(all(check.passed for check in checks))

    def test_gradient_check_rejects_empty_inputs_and_non_value_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            gradient_check(lambda _: Value(1.0), {})
        with self.assertRaisesRegex(TypeError, "return a Value"):
            gradient_check(lambda _: 1.0, {"x": 1.0})  # type: ignore[arg-type,return-value]


class ExperimentTests(unittest.TestCase):
    def test_tiny_neuron_requires_every_named_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing values"):
            tiny_neuron_loss({"x1": Value(1.0)})

    def test_experiment_is_deterministic_and_checks_all_inputs(self) -> None:
        first = run_autodiff_experiment()
        second = run_autodiff_experiment()
        self.assertEqual(first, second)
        self.assertEqual(
            [check.name for check in first.gradient_checks],
            list(default_experiment_inputs()),
        )
        self.assertTrue(all(check.passed for check in first.gradient_checks))

    def test_experiment_metrics_report_graph_and_error(self) -> None:
        metrics = experiment_metrics(run_autodiff_experiment())
        self.assertEqual(metrics["gradient_check_count"], 6)
        self.assertGreater(int(metrics["node_count"]), 6)
        self.assertGreater(int(metrics["edge_count"]), 6)
        self.assertLess(float(metrics["max_gradient_error"]), 1e-7)
        self.assertTrue(metrics["all_gradients_pass"])

    def test_validation_rejects_failed_gradient_check(self) -> None:
        experiment = run_autodiff_experiment()
        broken_check = replace(experiment.gradient_checks[0], passed=False)
        broken = AutodiffExperiment(
            experiment.output,
            experiment.graph,
            (broken_check, *experiment.gradient_checks[1:]),
        )
        with self.assertRaisesRegex(ValueError, "gradient checks failed"):
            validate_experiment(broken)

    def test_validation_rejects_missing_gradient_checks(self) -> None:
        experiment = run_autodiff_experiment()
        broken = AutodiffExperiment(experiment.output, experiment.graph, ())
        with self.assertRaisesRegex(ValueError, "must contain gradient checks"):
            validate_experiment(broken)

    def test_markdown_reports_metrics_graph_and_scope_limit(self) -> None:
        report = render_autodiff_markdown(run_autodiff_experiment())
        self.assertIn("# Day 15 scalar autodiff report", report)
        self.assertIn("maximum gradient-check error", report)
        self.assertIn("```mermaid", report)
        self.assertIn("does not claim lecture completion", report)


if __name__ == "__main__":
    unittest.main()
