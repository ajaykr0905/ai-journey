from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.manual_backprop import (
    ManualBackpropExperiment,
    ManualGraphError,
    NodeSpec,
    apply_operation,
    autodiff_gradients,
    build_tiny_graph,
    closed_form_gradients,
    evaluate_forward,
    experiment_metrics,
    gradient_map,
    local_derivatives,
    manual_backward,
    render_manual_backprop_markdown,
    render_trace_mermaid,
    run_manual_backprop_experiment,
    topological_order,
    validate_experiment,
    validate_graph,
)


class GraphValidationTests(unittest.TestCase):
    def test_empty_graph_is_rejected(self) -> None:
        with self.assertRaisesRegex(ManualGraphError, "at least one"):
            validate_graph((), "y")

    def test_root_must_exist(self) -> None:
        with self.assertRaisesRegex(ManualGraphError, "unknown root"):
            validate_graph((NodeSpec("x", "input", value=1.0),), "y")

    def test_duplicate_node_names_are_rejected(self) -> None:
        nodes = (
            NodeSpec("x", "input", value=1.0),
            NodeSpec("x", "input", value=2.0),
        )
        with self.assertRaisesRegex(ManualGraphError, "duplicate"):
            validate_graph(nodes, "x")

    def test_unsupported_operation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ManualGraphError, "unsupported operation"):
            validate_graph((NodeSpec("x", "sin"),), "x")

    def test_input_requires_finite_value_and_no_parents(self) -> None:
        for node in (
            NodeSpec("x", "input"),
            NodeSpec("x", "input", ("y",), 1.0),
            NodeSpec("x", "input", value=float("inf")),
        ):
            with (
                self.subTest(node=node),
                self.assertRaises((ManualGraphError, ValueError)),
            ):
                validate_graph((node,), "x")

    def test_operation_requires_two_parents_and_no_literal_value(self) -> None:
        base = NodeSpec("x", "input", value=1.0)
        for node in (
            NodeSpec("y", "add", ("x",)),
            NodeSpec("y", "mul", ("x", "x"), 2.0),
        ):
            with (
                self.subTest(node=node),
                self.assertRaisesRegex(ManualGraphError, "two parents|cannot declare"),
            ):
                validate_graph((base, node), "y")

    def test_unknown_parent_is_rejected(self) -> None:
        nodes = (
            NodeSpec("x", "input", value=1.0),
            NodeSpec("y", "add", ("x", "missing")),
        )
        with self.assertRaisesRegex(ManualGraphError, "unknown parents"):
            validate_graph(nodes, "y")

    def test_cycle_is_rejected(self) -> None:
        nodes = (
            NodeSpec("one", "input", value=1.0),
            NodeSpec("left", "add", ("right", "one")),
            NodeSpec("right", "mul", ("left", "one")),
        )
        with self.assertRaisesRegex(ManualGraphError, "acyclic"):
            validate_graph(nodes, "left")

    def test_topological_order_places_parents_before_results(self) -> None:
        nodes = build_tiny_graph()
        order = topological_order(nodes, "y")
        positions = {name: index for index, name in enumerate(order)}
        for node in nodes:
            if node.name not in positions:
                continue
            for parent in node.parents:
                self.assertLess(positions[parent], positions[node.name])


class LocalRuleTests(unittest.TestCase):
    def test_add_forward_and_local_derivatives(self) -> None:
        self.assertEqual(apply_operation("add", (2.0, -3.0)), -1.0)
        self.assertEqual(local_derivatives("add", (2.0, -3.0)), (1.0, 1.0))

    def test_mul_forward_and_local_derivatives(self) -> None:
        self.assertEqual(apply_operation("mul", (2.0, -3.0)), -6.0)
        self.assertEqual(local_derivatives("mul", (2.0, -3.0)), (-3.0, 2.0))

    def test_local_rules_require_two_finite_parents(self) -> None:
        for function in (apply_operation, local_derivatives):
            with (
                self.subTest(function=function.__name__),
                self.assertRaises((ManualGraphError, ValueError)),
            ):
                function("add", (1.0, float("nan")))
            with self.assertRaisesRegex(ManualGraphError, "two parent"):
                function("add", (1.0,))

    def test_unknown_local_rule_is_rejected(self) -> None:
        with self.assertRaisesRegex(ManualGraphError, "unsupported"):
            apply_operation("divide", (2.0, 1.0))
        with self.assertRaisesRegex(ManualGraphError, "unsupported"):
            local_derivatives("divide", (2.0, 1.0))


class ManualPassTests(unittest.TestCase):
    def test_forward_pass_records_every_expected_intermediate(self) -> None:
        steps = evaluate_forward(build_tiny_graph(), "y")
        values = {step.name: step.value for step in steps}
        self.assertEqual(
            values,
            {
                "a": 2.0,
                "b": -3.0,
                "product": -6.0,
                "c": 10.0,
                "shifted": 4.0,
                "d": -2.0,
                "scaled": -8.0,
                "y": -6.0,
            },
        )

    def test_manual_pass_matches_hand_calculated_gradients(self) -> None:
        trace = manual_backward(build_tiny_graph(), "y")
        gradients = gradient_map(trace)
        self.assertEqual(
            {name: gradients[name] for name in ("a", "b", "c", "d")},
            {"a": 7.0, "b": -4.0, "c": -2.0, "d": 4.0},
        )

    def test_non_unit_seed_scales_every_input_gradient(self) -> None:
        trace = manual_backward(build_tiny_graph(), "y", seed=0.25)
        gradients = gradient_map(trace)
        self.assertEqual(
            {name: gradients[name] for name in ("a", "b", "c", "d")},
            {"a": 1.75, "b": -1.0, "c": -0.5, "d": 1.0},
        )

    def test_shared_node_accumulates_one_contribution_per_edge(self) -> None:
        nodes = (
            NodeSpec("x", "input", value=3.0),
            NodeSpec("square", "mul", ("x", "x")),
        )
        trace = manual_backward(nodes, "square")
        x_steps = [step for step in trace.reverse_steps if step.parent == "x"]
        self.assertEqual(len(x_steps), 2)
        self.assertEqual([step.contribution for step in x_steps], [3.0, 3.0])
        self.assertEqual([step.gradient_after for step in x_steps], [3.0, 6.0])

    def test_reverse_trace_processes_results_before_their_inputs(self) -> None:
        trace = manual_backward(build_tiny_graph(), "y")
        processed_results = [step.result for step in trace.reverse_steps]
        self.assertLess(processed_results.index("y"), processed_results.index("scaled"))
        self.assertLess(
            processed_results.index("scaled"), processed_results.index("shifted")
        )

    def test_build_tiny_graph_requires_exact_named_inputs(self) -> None:
        with self.assertRaisesRegex(ManualGraphError, "exactly"):
            build_tiny_graph({"a": 1.0})

    def test_closed_form_derivatives_match_expanded_expression(self) -> None:
        gradients = closed_form_gradients({"a": 2.0, "b": -3.0, "c": 10.0, "d": -2.0})
        self.assertEqual(gradients, {"a": 7.0, "b": -4.0, "c": -2.0, "d": 4.0})

    def test_separate_autodiff_engine_matches_manual_result(self) -> None:
        gradients = autodiff_gradients({"a": 2.0, "b": -3.0, "c": 10.0, "d": -2.0})
        self.assertEqual(gradients, {"a": 7.0, "b": -4.0, "c": -2.0, "d": 4.0})


class ExperimentTests(unittest.TestCase):
    def test_experiment_is_deterministic_and_compares_all_inputs(self) -> None:
        first = run_manual_backprop_experiment()
        second = run_manual_backprop_experiment()
        self.assertEqual(first, second)
        self.assertEqual(
            [item.name for item in first.comparisons], ["a", "b", "c", "d"]
        )
        self.assertTrue(all(item.passed for item in first.comparisons))

    def test_metrics_capture_trace_size_accumulation_and_accuracy(self) -> None:
        metrics = experiment_metrics(run_manual_backprop_experiment())
        self.assertEqual(metrics["node_count"], 8)
        self.assertEqual(metrics["reverse_step_count"], 8)
        self.assertEqual(metrics["comparison_count"], 4)
        self.assertEqual(metrics["accumulation_step_count"], 1)
        self.assertEqual(metrics["output"], -6.0)
        self.assertEqual(metrics["max_comparison_error"], 0.0)
        self.assertTrue(metrics["all_comparisons_pass"])

    def test_validation_rejects_a_failed_comparison(self) -> None:
        experiment = run_manual_backprop_experiment()
        broken_item = replace(experiment.comparisons[0], passed=False)
        broken = ManualBackpropExperiment(
            experiment.nodes,
            experiment.trace,
            (broken_item, *experiment.comparisons[1:]),
        )
        with self.assertRaisesRegex(ManualGraphError, "comparisons failed"):
            validate_experiment(broken)

    def test_validation_rejects_missing_input_comparison(self) -> None:
        experiment = run_manual_backprop_experiment()
        broken = ManualBackpropExperiment(
            experiment.nodes, experiment.trace, experiment.comparisons[:-1]
        )
        with self.assertRaisesRegex(ManualGraphError, "all four"):
            validate_experiment(broken)

    def test_mermaid_labels_values_gradients_and_local_derivatives(self) -> None:
        diagram = render_trace_mermaid(run_manual_backprop_experiment())
        self.assertIn("flowchart LR", diagram)
        self.assertIn("a<br/>value=2.000<br/>grad=7.000", diagram)
        self.assertIn('a -->|"local=-3.000"| product', diagram)

    def test_markdown_contains_auditable_trace_and_scope_limit(self) -> None:
        report = render_manual_backprop_markdown(run_manual_backprop_experiment())
        self.assertIn("# Day 16 manual backpropagation report", report)
        self.assertIn("Reverse chain-rule trace", report)
        self.assertIn("Independent gradient comparisons", report)
        self.assertIn("does not claim lecture completion", report)


if __name__ == "__main__":
    unittest.main()
