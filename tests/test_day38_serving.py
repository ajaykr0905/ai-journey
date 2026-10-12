"""Serving accepts only models bound to complete verified training artifacts."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from ai_journey.day38_training import (
    Day38DataConfig,
    Day38TrainingConfig,
    build_day38_report,
    run_day38_experiment,
    write_day38_report,
)
from ai_journey.day38_serving import load_trained_artifact
from ai_journey.gpt_bigram import CorpusSource


class Day38ServingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        text = "First Citizen:\nWe know the model.\n" * 20
        source = CorpusSource(
            "test",
            "https://example.test/corpus",
            hashlib.sha256(text.encode()).hexdigest(),
            len(text.encode()),
            "test-only",
        )
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text(text)
        self.checkpoint = self.root / "checkpoint.pt"
        self.report = self.root / "report.json"
        config = Day38TrainingConfig(
            data=Day38DataConfig(
                block_size=4,
                train_token_limit=320,
                validation_token_limit=80,
                validation_fraction=0.2,
            ),
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            steps=2,
            batch_size=2,
            sample_tokens=8,
            sample_top_k=5,
        )
        experiment = run_day38_experiment(
            text, source, config, checkpoint_path=self.checkpoint
        )
        write_day38_report(
            self.report, build_day38_report(experiment, elapsed_seconds=1)
        )

    def load(self):
        return load_trained_artifact(self.report, self.checkpoint, self.corpus)

    def test_read_only_load_binds_weights_and_preserves_rng(self):
        checkpoint = self.checkpoint.read_bytes()
        rng = torch.get_rng_state().clone()
        artifact = self.load()
        self.assertFalse(artifact.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(checkpoint, self.checkpoint.read_bytes())

    def test_wrong_source_or_oversized_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            load_trained_artifact(
                self.report, self.checkpoint, self.corpus, max_parameters=1
            )
        self.corpus.write_text("changed source")
        with self.assertRaises(ValueError):
            self.load()

    def test_changed_checkpoint_weights_are_rejected(self):
        payload = torch.load(self.checkpoint, weights_only=True)
        payload["model_state"]["lm_head.weight"][0, 0] += 0.1
        torch.save(payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.load()
