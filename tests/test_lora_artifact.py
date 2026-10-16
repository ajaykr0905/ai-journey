from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from torch import nn

from ai_journey.lora import LoRAConfig, inject_adapters
from ai_journey.lora_artifact import write_adapter_artifact


class ArtifactWriterTests(unittest.TestCase):
    def setUp(self):
        self.model = inject_adapters(
            nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig()
        )

    def test_artifact_is_deterministic_json_with_a_verifiable_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.json"
            digest = write_adapter_artifact(path, self.model)
            first = path.read_bytes()
            self.assertEqual(write_adapter_artifact(path, self.model), digest)
            self.assertEqual(path.read_bytes(), first)
            payload = json.loads(first)
            self.assertEqual(payload.pop("sha256"), digest)
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            self.assertEqual(sha256(canonical).hexdigest(), digest)
            self.assertEqual(set(payload["layers"]["0"]), {"config", "a", "b"})

    def test_replace_failure_preserves_old_artifact_and_removes_owned_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.json"
            write_adapter_artifact(path, self.model)
            before = path.read_bytes()
            with patch(
                "ai_journey.lora_artifact.os.replace", side_effect=OSError("fixture")
            ):
                with self.assertRaises(OSError):
                    write_adapter_artifact(path, self.model)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_serialization_budget_is_checked_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent" / "adapter.json"
            with patch("ai_journey.lora_artifact.MAX_ARTIFACT_BYTES", 10):
                with self.assertRaises(ValueError):
                    write_adapter_artifact(path, self.model)
            self.assertFalse(path.parent.exists())
