from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ai_journey.bpe_artifact import load_bpe_artifact
from ai_journey.bpe_train_cli import main
from ai_journey.bpe_serving import generate_utf8


class BPEArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = [
            root / name
            for name in ("report", "checkpoint", "tokenizer", "train", "validation")
        ]
        self.paths[3].write_text("hello hello world\nनमस्ते नमस्ते\n", encoding="utf-8")
        self.paths[4].write_text("held out Unicode 🙂\n", encoding="utf-8")
        args = [
            item
            for name, path in zip(
                ("output", "checkpoint", "tokenizer", "train", "validation"), self.paths
            )
            for item in (f"--{name}", str(path))
        ]
        with redirect_stdout(io.StringIO()):
            main(args + ["--steps", "1", "--block-size", "4", "--vocab-size", "270"])

    def test_verified_weights_can_serve_without_mutating_caller_rng(self):
        before = torch.get_rng_state().clone()
        artifact = load_bpe_artifact(*self.paths)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        result = generate_utf8(
            artifact.model, artifact.corpus.codec, "hello", new_tokens=4
        )
        result["text"].encode("utf-8")
        self.assertFalse(artifact.model.training)

    def test_changed_codec_data_or_reported_loss_is_rejected(self):
        report = json.loads(self.paths[0].read_text())
        report["experiment"]["final_validation_nll"] += 1
        self.paths[0].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "reported loss"):
            load_bpe_artifact(*self.paths)
        self.paths[4].write_text("changed public corpus")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_bpe_artifact(*self.paths)

    def test_oversized_report_is_rejected_before_model_allocation(self):
        report = json.loads(self.paths[0].read_text())
        report["experiment"]["model_config"]["embedding_dim"] = 10000
        self.paths[0].write_text(json.dumps(report))
        with patch(
            "ai_journey.bpe_artifact.DecoderLanguageModel"
        ) as allocation, self.assertRaisesRegex(ValueError, "budget"):
            load_bpe_artifact(*self.paths)
        allocation.assert_not_called()
