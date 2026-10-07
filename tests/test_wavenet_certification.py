from __future__ import annotations

import unittest

import torch

from ai_journey.wavenet import WaveNetConfig, initialize_wavenet
from ai_journey.wavenet_certification import (
    audit_parameter_manifest,
    audit_storage_independence,
    parameter_inventory,
)
from ai_journey.wavenet_rebuild import initialize_rebuilt_wavenet


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


if __name__ == "__main__":
    unittest.main()
