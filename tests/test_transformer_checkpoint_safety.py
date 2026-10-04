"""Checkpoint publication must preserve the previous recoverable state."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    load_training_checkpoint,
    model_fingerprint,
    save_training_checkpoint,
)


class CheckpointPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_rng = torch.get_rng_state().clone()
        self.training = TrainingConfig(steps=3, batch_size=2)
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.cursor = BatchCursor(torch.arange(30) % 5, block_size=4, batch_size=2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(torch.set_rng_state, self.original_rng)
        self.path = Path(self.directory.name) / "checkpoint.pt"

    def save(self, step: int = 1) -> None:
        save_training_checkpoint(
            self.path,
            model=self.model,
            optimizer=self.optimizer,
            cursor=self.cursor,
            training_config=self.training,
            corpus_fingerprint="a" * 64,
            step=step,
        )

    def test_success_syncs_serialized_state_before_replacement(self) -> None:
        with patch("os.fsync", wraps=os.fsync) as sync:
            self.save()
        self.assertGreaterEqual(sync.call_count, 1)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            ),
            1,
        )

    def test_serialization_failure_preserves_checkpoint_and_removes_partial_file(self):
        self.save()
        original = self.path.read_bytes()

        def fail_mid_write(payload, stream):
            if isinstance(stream, Path):
                stream.write_bytes(b"partial archive")
            else:
                stream.write(b"partial archive")
            raise OSError("simulated serialization failure")

        with patch("torch.save", side_effect=fail_mid_write):
            with self.assertRaisesRegex(OSError, "serialization failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_sync_failure_keeps_previous_checkpoint(self) -> None:
        self.save()
        original = self.path.read_bytes()
        with patch("os.fsync", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_replacement_failure_keeps_previous_checkpoint(self) -> None:
        self.save()
        original = self.path.read_bytes()
        with patch("os.replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaisesRegex(OSError, "replacement failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_concurrent_writers_publish_complete_independent_archives(self) -> None:
        barrier = threading.Barrier(2, timeout=10)
        real_save = torch.save

        def synchronize_writes(payload, stream):
            real_save(payload, stream)
            barrier.wait()

        with patch("torch.save", side_effect=synchronize_writes):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.save, step) for step in (1, 2)]
                for future in futures:
                    future.result(timeout=15)
        payload = torch.load(self.path, weights_only=True)
        self.assertIn(payload["step"], (1, 2))
        self.assertEqual(payload["model_fingerprint"], model_fingerprint(self.model))
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
