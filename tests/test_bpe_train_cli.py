from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from ai_journey.bpe_train_cli import main
from ai_journey.tokenizer_io import load_tokenizer, tokenizer_fingerprint


class BPETrainingCliTests(unittest.TestCase):
    def test_training_and_completed_resume_publish_matching_codec_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = root / "train", root / "validation"
            train.write_text("hello hello world\nनमस्ते नमस्ते दुनिया\n", encoding="utf-8")
            validation.write_text("held out Unicode 🙂\n", encoding="utf-8")
            args = [
                "--train",
                str(train),
                "--validation",
                str(validation),
                "--checkpoint",
                str(root / "model.pt"),
                "--tokenizer",
                str(root / "tokenizer.json"),
                "--output",
                str(root / "report.json"),
                "--steps",
                "2",
                "--vocab-size",
                "272",
                "--block-size",
                "4",
            ]
            with redirect_stdout(io.StringIO()):
                main(args)
            report = json.loads((root / "report.json").read_text())["experiment"]
            self.assertEqual(
                report["codec_fingerprint"],
                tokenizer_fingerprint(load_tokenizer(root / "tokenizer.json")),
            )
            with redirect_stdout(io.StringIO()):
                main(args + ["--resume"])
            resumed = json.loads((root / "report.json").read_text())["experiment"]
            self.assertEqual(resumed["model_fingerprint"], report["model_fingerprint"])
            self.assertEqual(resumed["trace"], [])

    def test_excessive_control_budget_fails_without_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = [
                value
                for name in ("train", "validation", "checkpoint", "tokenizer", "output")
                for value in (f"--{name}", str(root / name))
            ]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--steps", "101"])
            self.assertFalse((root / "checkpoint").exists())
