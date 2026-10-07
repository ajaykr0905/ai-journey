from __future__ import annotations

import unittest

import torch

from ai_journey.wavenet import WaveNetConfig, initialize_wavenet
from ai_journey.wavenet_certification import (
    audit_activation_finiteness,
    audit_eval_batch_invariance,
    audit_gradient_clipping,
    audit_gradient_coverage,
    audit_parameter_finiteness,
    audit_parameter_manifest,
    audit_per_example_loss_parity,
    audit_probability_simplex,
    audit_storage_independence,
    audit_top_k_parity,
    measure_activation_saturation,
    measure_gradient_cosines,
    measure_gradient_statistics,
    parameter_inventory,
    parameter_statistics,
    trace_rebuild_activations,
)
from ai_journey.wavenet_rebuild import (
    initialize_rebuilt_wavenet,
    load_reference_parameters,
)


class WaveNetCertificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WaveNetConfig(
            vocab_size=7,
            context_size=4,
            embedding_dim=3,
            hidden_dim=6,
            group_factors=(2, 2),
            dropout=0.0,
        )
        self.model = initialize_rebuilt_wavenet(self.config, seed=330)

    def test_parameter_inventory_covers_registered_trainable_tensors(self) -> None:
        entries = parameter_inventory(self.model)

        self.assertEqual(
            tuple(entry.name for entry in entries),
            tuple(name for name, _ in self.model.named_parameters()),
        )
        self.assertEqual(
            sum(entry.elements for entry in entries), self.model.parameter_count
        )
        self.assertTrue(all(entry.dtype == str(torch.float32) for entry in entries))
        self.assertTrue(all(entry.requires_gradient for entry in entries))

    def test_parameter_inventory_rejects_other_models(self) -> None:
        with self.assertRaises(TypeError):
            parameter_inventory(torch.nn.Linear(2, 2))  # type: ignore[arg-type]

    def test_parameter_manifest_matches_the_compiled_plan(self) -> None:
        audit = audit_parameter_manifest(self.model)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.registered_elements, self.model.parameter_count)
        self.assertEqual(audit.planned_elements, self.model.plan.parameter_count)
        self.assertGreater(audit.tensors, 0)

    def test_reference_and_rebuild_do_not_share_parameter_storage(self) -> None:
        reference = initialize_wavenet(self.config, seed=330)

        audit = audit_storage_independence(reference, self.model)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.reference_tensors, audit.rebuild_tensors)

    def test_parameter_finiteness_identifies_the_corrupt_tensor(self) -> None:
        self.assertTrue(audit_parameter_finiteness(self.model).passed)
        with torch.no_grad():
            self.model.output_bias[0] = float("nan")

        audit = audit_parameter_finiteness(self.model)

        self.assertFalse(audit.passed)
        self.assertEqual(audit.nonfinite_parameters, ("output_bias",))

    def test_parameter_statistics_cover_each_registered_tensor(self) -> None:
        statistics = parameter_statistics(self.model)

        self.assertEqual(
            tuple(item.name for item in statistics),
            tuple(name for name, _ in self.model.named_parameters()),
        )
        self.assertTrue(all(item.l2_norm >= 0 for item in statistics))
        self.assertTrue(all(item.minimum <= item.maximum for item in statistics))

    def test_activation_trace_covers_each_primitive_boundary(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        self.model.train()

        snapshots = trace_rebuild_activations(self.model, contexts)

        self.assertEqual(
            tuple(item.name for item in snapshots),
            ("embedding", "stage_1", "stage_2", "logits"),
        )
        self.assertEqual(snapshots[-1].shape, (2, self.config.vocab_size))
        self.assertTrue(self.model.training)
        self.assertTrue(all(len(item.digest) == 64 for item in snapshots))

    def test_activation_finiteness_locates_propagated_corruption(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(audit_activation_finiteness(self.model, contexts).passed)
        with torch.no_grad():
            self.model.stage_weights[0][0, 0] = float("inf")

        audit = audit_activation_finiteness(self.model, contexts)

        self.assertFalse(audit.passed)
        self.assertEqual(audit.nonfinite_boundaries, ("stage_1", "stage_2", "logits"))

    def test_activation_saturation_is_measured_per_tanh_stage(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

        measurements = measure_activation_saturation(
            self.model, contexts, threshold=0.9
        )

        self.assertEqual(
            tuple(item.stage for item in measurements), ("stage_1", "stage_2")
        )
        self.assertTrue(all(0 <= item.saturated_fraction <= 1 for item in measurements))
        self.assertTrue(all(item.elements > 0 for item in measurements))
        with self.assertRaises(ValueError):
            measure_activation_saturation(self.model, contexts, threshold=0.0)

    def test_eval_prediction_is_invariant_to_batch_neighbors(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3], [2, 2, 2, 2]])
        self.model.train()

        audit = audit_eval_batch_invariance(self.model, contexts)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.batch_size, 3)
        self.assertTrue(self.model.training)

    def test_reference_and_rebuild_rank_the_same_top_k_tokens(self) -> None:
        reference = initialize_wavenet(self.config, seed=331)
        load_reference_parameters(self.model, reference)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])

        audit = audit_top_k_parity(reference, self.model, contexts, k=3)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.examples, 2)
        self.assertEqual(audit.k, 3)

    def test_reference_and_rebuild_match_each_example_loss(self) -> None:
        reference = initialize_wavenet(self.config, seed=332)
        load_reference_parameters(self.model, reference)
        contexts = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
        targets = torch.tensor([4, 0])

        audit = audit_per_example_loss_parity(reference, self.model, contexts, targets)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.examples, 2)

    def test_prediction_probabilities_form_finite_simplex_rows(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3]])

        audit = audit_probability_simplex(self.model, contexts)

        self.assertTrue(audit.passed)
        self.assertEqual(audit.examples, 2)
        self.assertLessEqual(audit.max_row_sum_error, 1e-6)

    def test_gradient_statistics_cover_parameters_without_mutating_gradients(
        self,
    ) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3]])
        targets = torch.tensor([4, 2])
        self.model.output_bias.grad = torch.ones_like(self.model.output_bias)

        statistics = measure_gradient_statistics(self.model, contexts, targets)

        self.assertEqual(len(statistics), len(tuple(self.model.parameters())))
        self.assertTrue(all(item.finite for item in statistics))
        self.assertTrue(all(item.l2_norm >= 0 for item in statistics))
        self.assertTrue(torch.equal(self.model.output_bias.grad, torch.ones(7)))

    def test_mapped_gradient_directions_are_identical(self) -> None:
        reference = initialize_wavenet(self.config, seed=333)
        load_reference_parameters(self.model, reference)
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3]])
        targets = torch.tensor([4, 2])

        measurements = measure_gradient_cosines(
            reference, self.model, contexts, targets
        )

        self.assertEqual(len(measurements), len(tuple(self.model.parameters())))
        self.assertTrue(all(item.cosine_similarity > 0.999999 for item in measurements))

    def test_gradient_coverage_detects_a_frozen_registered_parameter(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3]])
        targets = torch.tensor([4, 2])
        self.assertTrue(audit_gradient_coverage(self.model, contexts, targets).passed)
        self.model.output_bias.requires_grad_(False)

        audit = audit_gradient_coverage(self.model, contexts, targets)

        self.assertFalse(audit.passed)
        self.assertEqual(audit.missing_gradients, ("output_bias",))

    def test_gradient_clipping_enforces_the_declared_global_norm(self) -> None:
        contexts = torch.tensor([[0, 1, 2, 3], [6, 5, 4, 3]])
        targets = torch.tensor([4, 2])

        audit = audit_gradient_clipping(self.model, contexts, targets, max_norm=0.01)

        self.assertTrue(audit.passed)
        self.assertTrue(audit.clipped)
        self.assertLessEqual(audit.post_clip_norm, 0.010001)


if __name__ == "__main__":
    unittest.main()
