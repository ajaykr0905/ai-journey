from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ai_journey.lora import LoRAConfig, LoRAError, inject_adapters
from ai_journey.lora_artifact import (
    write_adapter_artifact,
    read_adapter_artifact,
    restore_adapter_artifact,
)


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


class ArtifactReaderTests(unittest.TestCase):
    def setUp(self):
        self.base = nn.Sequential(nn.Linear(4, 4))
        self.model = inject_adapters(self.base, ("0",), LoRAConfig())

    def test_round_trip_restores_predictions_for_float32_float64_and_bfloat16(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                base = self.base.to(dtype)
                first = inject_adapters(base, ("0",), LoRAConfig())
                second = inject_adapters(base, ("0",), LoRAConfig())
                with torch.no_grad():
                    first[0].lora_b.fill_(0.25)
                path = Path(directory) / "adapter.json"
                write_adapter_artifact(path, first)
                restore_adapter_artifact(path, second)
                x = torch.ones(2, 4, dtype=dtype)
                self.assertTrue(torch.equal(first(x), second(x)))

    def test_corrupt_or_ambiguous_json_never_changes_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.json"
            write_adapter_artifact(path, self.model)
            raw = path.read_text()
            for candidate in (
                raw.replace('"rank":2', '"rank":1'),
                '{"schema":1,"schema":2}',
                '{"bad":NaN}',
                "[]",
                "{",
            ):
                path.write_text(candidate)
                with self.assertRaises(LoRAError):
                    restore_adapter_artifact(path, self.model)
                self.assertEqual(float(self.model[0].lora_b.detach().abs().sum()), 0)

    def test_recomputed_digest_does_not_bypass_shape_or_numeric_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.json"
            write_adapter_artifact(path, self.model)
            original = json.loads(path.read_bytes())
            for key, value in (
                ("shape", [65536, 65536]),
                ("values", [True] * 8),
                ("dtype", "torch.int64"),
            ):
                payload = json.loads(json.dumps(original))
                payload["layers"]["0"]["a"][key] = value
                payload.pop("sha256")
                canonical = json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
                payload["sha256"] = sha256(canonical).hexdigest()
                path.write_text(json.dumps(payload))
                with self.assertRaises(LoRAError):
                    read_adapter_artifact(path)

    def test_byte_budget_and_wrong_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.json"
            write_adapter_artifact(path, self.model)
            with patch("ai_journey.lora_artifact.MAX_ARTIFACT_BYTES", 16):
                with self.assertRaises(LoRAError):
                    read_adapter_artifact(path)
            wrong = inject_adapters(
                nn.Sequential(nn.Linear(4, 4)), ("0",), LoRAConfig()
            )
            with self.assertRaises(LoRAError):
                restore_adapter_artifact(path, wrong)
