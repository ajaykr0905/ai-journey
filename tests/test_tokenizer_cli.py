from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from ai_journey.tokenizer_cli import main


class TokenizerCliTests(unittest.TestCase):
    def test_training_roundtrip_and_audit_are_operable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, checkpoint, audit = (
                root / name for name in ("corpus", "tokenizer", "audit")
            )
            corpus.write_text("hello hello\nनमस्ते नमस्ते\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "train",
                            "--corpus",
                            str(corpus),
                            "--output",
                            str(checkpoint),
                            "--vocab-size",
                            "280",
                            "--regex",
                        ]
                    ),
                    0,
                )
            output = io.StringIO()
            with redirect_stdout(output):
                main(["encode", "--tokenizer", str(checkpoint), "--text", "नमस्ते hello"])
            ids = output.getvalue()
            output = io.StringIO()
            with redirect_stdout(output):
                main(["decode", "--tokenizer", str(checkpoint), "--ids", ids])
            self.assertEqual(output.getvalue().strip(), "नमस्ते hello")
            main(
                [
                    "audit",
                    "--tokenizer",
                    str(checkpoint),
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(audit),
                ]
            )
            self.assertEqual(json.loads(audit.read_text())["metrics"]["documents"], 2)

    def test_training_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus"
            path.write_text("hello hello")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["train", "--corpus", str(path), "--output", str(path)])
            self.assertEqual(path.read_text(), "hello hello")
