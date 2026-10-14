from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
)


class CheckpointArchiveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5, block_size=2, embedding_dim=4, head_count=1, layer_count=1
            )
        )
        self.config = TrainingConfig(steps=1, batch_size=1)
        self.optimizer = build_optimizer(self.model, self.config)
        self.cursor = BatchCursor(torch.arange(10) % 5, block_size=2, batch_size=1)

    def controls(self):
        return dict(
            model=self.model,
            optimizer=self.optimizer,
            cursor=self.cursor,
            training_config=self.config,
            corpus_fingerprint="a" * 64,
        )

    def test_raw_size_and_compressed_expansion_fail_before_tensor_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.write_bytes(b"x" * 2048)
            with patch(
                "ai_journey.transformer_lab.torch.load"
            ) as deserialize, self.assertRaises(ValueError):
                load_training_checkpoint(
                    path, max_checkpoint_bytes=1024, **self.controls()
                )
            deserialize.assert_not_called()
            with zipfile.ZipFile(
                path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr("archive/data/0", b"x" * 2 * 1024 * 1024)
            self.assertLess(path.stat().st_size, 1024 * 1024)
            with patch(
                "ai_journey.transformer_lab.torch.load"
            ) as deserialize, self.assertRaises(ValueError):
                load_training_checkpoint(
                    path, max_checkpoint_bytes=1024 * 1024, **self.controls()
                )
            deserialize.assert_not_called()

    def test_normal_lab_archive_still_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            save_training_checkpoint(path, step=0, **self.controls())
            self.assertEqual(load_training_checkpoint(path, **self.controls()), 0)
