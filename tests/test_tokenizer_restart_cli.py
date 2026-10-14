from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from ai_journey.bpe import TokenizerError
from ai_journey.tokenizer_io import load_tokenizer
from ai_journey.tokenizer_restart_cli import main, load_training_state
from ai_journey.tokenizer_training import train_documents


class DurableTrainingRestartTests(unittest.TestCase):
    def test_two_process_invocations_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, tokenizer, state = (
                root / name for name in ("corpus", "tokenizer", "state")
            )
            documents = ("banana banana hello", "नमस्ते नमस्ते दुनिया")
            corpus.write_text("\n".join(documents), encoding="utf-8")
            args = [
                "--corpus",
                str(corpus),
                "--tokenizer",
                str(tokenizer),
                "--state",
                str(state),
            ]
            with redirect_stdout(io.StringIO()):
                main(args + ["--vocab-size", "262"])
                main(args + ["--resume", "--vocab-size", "280"])
            self.assertEqual(
                load_tokenizer(tokenizer).merges,
                train_documents(documents, vocab_size=280).merges,
            )
            before = state.read_bytes()
            corpus.write_text("changed public input")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ["--resume"])
            self.assertEqual(state.read_bytes(), before)

    def test_corrupted_digest_is_rejected_before_training_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            path.write_text(
                json.dumps({"schema_version": 1, "state": {}, "sha256": "a" * 64})
            )
            with self.assertRaisesRegex(TokenizerError, "digest"):
                load_training_state(path, ("public",))
