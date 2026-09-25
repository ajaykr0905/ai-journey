from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_checkpoint import checkpoint_payload
from ai_journey.context_mlp import build_context_dataset, initialize_context_mlp


class ContextCheckpointTests(unittest.TestCase):
    def test_payload_is_json_compatible_and_versioned(self) -> None:
        dataset = build_context_dataset(("anna", "aria"))
        model = initialize_context_mlp(dataset, seed=7)
        payload = checkpoint_payload(model, step=12)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["step"], 12)
        self.assertEqual(len(payload["model_fingerprint"]), 64)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
