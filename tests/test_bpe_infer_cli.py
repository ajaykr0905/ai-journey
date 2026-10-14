from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from ai_journey.bpe_infer_cli import main
from ai_journey.bpe_train_cli import main as train


class BPEInferenceCliTests(unittest.TestCase):
    def test_trained_samples_reproduce_exactly_and_preserve_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").write_text(
                "hello hello public text\nनमस्ते नमस्ते दुनिया\n", encoding="utf-8"
            )
            (root / "validation").write_text("held out Unicode 🙂\n", encoding="utf-8")
            paths = {
                name: root / name
                for name in ("report", "checkpoint", "tokenizer", "train", "validation")
            }
            training_args = [
                item
                for name, path in paths.items()
                for item in (f"--{'output' if name == 'report' else name}", str(path))
            ]
            with redirect_stdout(io.StringIO()):
                train(training_args + ["--steps", "1", "--block-size", "4"])
            before = {name: path.read_bytes() for name, path in paths.items()}
            args = [
                item
                for name, path in paths.items()
                for item in (f"--{name}", str(path))
            ] + [
                "--output",
                str(root / "sample"),
                "--prompt",
                "hello",
                "--new-tokens",
                "4",
            ]
            with redirect_stdout(io.StringIO()):
                main(args)
                first = json.loads((root / "sample").read_text())
                main(args)
            self.assertEqual(first, json.loads((root / "sample").read_text()))
            self.assertTrue(first["sample"]["text"].startswith("hello"))
            self.assertEqual(
                before, {name: path.read_bytes() for name, path in paths.items()}
            )

    def test_output_alias_cannot_overwrite_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = [
                item
                for name in ("report", "checkpoint", "tokenizer", "train", "validation")
                for item in (f"--{name}", str(root / name))
            ]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--output", str(root / "train"), "--prompt", "hello"])
