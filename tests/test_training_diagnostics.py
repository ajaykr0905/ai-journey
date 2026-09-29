from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.training_diagnostics import (
    ActivationCollector,
    DiagnosticError,
    HistogramSpec,
    capture_training_snapshot,
    collect_parameter_gradients,
    summarize_tensor,
)


class HistogramSpecTests(unittest.TestCase):
    def test_spec_rejects_invalid_ranges_and_counts(self) -> None:
        with self.assertRaisesRegex(DiagnosticError, "less than"):
            HistogramSpec(lower=1, upper=1)
        with self.assertRaisesRegex(DiagnosticError, "positive"):
            HistogramSpec(bins=0)
        with self.assertRaisesRegex(DiagnosticError, "finite"):
            HistogramSpec(upper=math.inf)


class TensorDistributionTests(unittest.TestCase):
    def test_summary_tracks_nonfinite_and_out_of_range_values(self) -> None:
        values = torch.tensor([float("-inf"), -2.0, -0.5, 0.0, 0.5, 2.0, float("nan")])
        summary = summarize_tensor(
            values, HistogramSpec(lower=-1, upper=1, bins=2, near_zero=0.1)
        )
        self.assertEqual(summary.count, 7)
        self.assertEqual(summary.finite_count, 5)
        self.assertEqual(summary.nonfinite_count, 2)
        self.assertEqual(summary.underflow_count, 1)
        self.assertEqual(summary.overflow_count, 1)
        self.assertEqual(sum(summary.histogram_counts), 3)
        self.assertEqual(summary.histogram_edges, (-1.0, 0.0, 1.0))
        self.assertAlmostEqual(summary.mean, 0.0)
        self.assertAlmostEqual(summary.zero_fraction, 0.2)
        self.assertAlmostEqual(summary.near_zero_fraction, 0.2)

    def test_summary_is_deterministic_and_json_serializable(self) -> None:
        values = torch.linspace(-2, 2, 17)
        first = summarize_tensor(values, HistogramSpec(bins=8)).to_dict()
        second = summarize_tensor(values, HistogramSpec(bins=8)).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(len(first["histogram_counts"]), 8)

    def test_summary_rejects_empty_or_entirely_nonfinite_tensors(self) -> None:
        with self.assertRaisesRegex(DiagnosticError, "empty"):
            summarize_tensor(torch.tensor([]))
        with self.assertRaisesRegex(DiagnosticError, "no finite"):
            summarize_tensor(torch.tensor([float("nan")]))


class ActivationCollectorTests(unittest.TestCase):
    def test_collector_records_selected_outputs_and_removes_hooks(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        )
        inputs = torch.ones(5, 3)
        with ActivationCollector(model, ("0", "1")) as collector:
            model(inputs)
            snapshot = collector.snapshot()
        self.assertEqual(snapshot["0"][0].count, 20)
        self.assertEqual(snapshot["1"][0].count, 20)
        model(inputs)
        self.assertEqual(collector.snapshot(), snapshot)

    def test_collector_clear_discards_prior_summaries(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        with ActivationCollector(model, ("1",)) as collector:
            model(torch.ones(1, 2))
            collector.clear()
            self.assertEqual(collector.snapshot(), {"1": ()})

    def test_collector_rejects_unknown_or_duplicate_modules(self) -> None:
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(DiagnosticError, "unknown"):
            ActivationCollector(model, ("missing",))
        with self.assertRaisesRegex(DiagnosticError, "unique"):
            ActivationCollector(model, ("", ""))


class ParameterGradientSnapshotTests(unittest.TestCase):
    def test_snapshot_reports_gradients_and_disconnected_parameters(self) -> None:
        class PartialModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                self.unused = torch.nn.Parameter(torch.tensor([3.0]))

            def forward(self) -> torch.Tensor:
                return self.used.square().sum()

        model = PartialModel()
        model().backward()
        snapshot = collect_parameter_gradients(
            model, HistogramSpec(lower=-5, upper=5, bins=10)
        )
        self.assertEqual([item.name for item in snapshot.gradients], ["used"])
        self.assertEqual(snapshot.missing_parameters, ("unused",))
        self.assertEqual(snapshot.gradients[0].distribution.count, 2)
        self.assertEqual(snapshot.to_dict()["missing_parameters"], ["unused"])

    def test_snapshot_is_empty_before_backward(self) -> None:
        model = torch.nn.Linear(2, 1)
        snapshot = collect_parameter_gradients(model)
        self.assertEqual(snapshot.gradients, ())
        self.assertEqual(snapshot.missing_parameters, ("bias", "weight"))


class TrainingSnapshotTests(unittest.TestCase):
    def test_capture_restores_mode_clears_gradients_and_preserves_weights(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(3, 2)
                self.activation = torch.nn.Tanh()

            def forward(
                self, inputs: torch.Tensor, targets: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                outputs = self.activation(self.projection(inputs))
                return outputs, torch.nn.functional.mse_loss(outputs, targets)

        torch.manual_seed(25)
        model = TinyModel().eval()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        snapshot = capture_training_snapshot(
            model,
            torch.ones(4, 3),
            torch.zeros(4, 2),
            ("projection", "activation"),
        )
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(
            [item.name for item in snapshot.activations],
            ["projection", "activation"],
        )
        self.assertEqual(snapshot.parameter_gradients.missing_parameters, ())
        self.assertGreater(snapshot.loss, 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_capture_requires_scalar_loss_contract(self) -> None:
        class InvalidModel(torch.nn.Module):
            def forward(
                self, inputs: torch.Tensor, targets: torch.Tensor
            ) -> torch.Tensor:
                return inputs + targets

        model = InvalidModel()
        with self.assertRaisesRegex(DiagnosticError, "scalar loss"):
            capture_training_snapshot(model, torch.ones(2), torch.ones(2), ("",))


if __name__ == "__main__":
    unittest.main()
