from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.context_checkpoint import (
    checkpoint_payload,
    load_checkpoint,
    model_from_payload,
    save_checkpoint,
)
from ai_journey.context_mlp import (
    build_context_dataset,
    initialize_context_mlp,
    model_fingerprint,
)


class ContextCheckpointTests(unittest.TestCase):
    def test_payload_is_json_compatible_and_versioned(self) -> None:
        dataset = build_context_dataset(("anna", "aria"))
        model = initialize_context_mlp(dataset, seed=7)
        payload = checkpoint_payload(model, step=12)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["step"], 12)
        self.assertEqual(len(payload["model_fingerprint"]), 64)
        json.dumps(payload)
        restored = model_from_payload(payload)
        self.assertEqual(model_fingerprint(restored), model_fingerprint(model))

    def test_tampered_payload_is_rejected(self) -> None:
        dataset = build_context_dataset(("anna", "aria"))
        payload = checkpoint_payload(initialize_context_mlp(dataset), step=0)
        payload["parameters"]["output_bias"][0] = 99
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            model_from_payload(payload)

    def test_save_checkpoint_writes_complete_json(self) -> None:
        dataset = build_context_dataset(("anna", "aria"))
        model = initialize_context_mlp(dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.json"
            save_checkpoint(path, model, step=4)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 4)
            self.assertFalse(path.with_name(".model.json.tmp").exists())
            restored, step = load_checkpoint(path)
            self.assertEqual(step, 4)
            self.assertEqual(model_fingerprint(restored), model_fingerprint(model))


if __name__ == "__main__":
    unittest.main()
