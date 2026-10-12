from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.day38_cli import main  # noqa: E402
from ai_journey.day38_training import verify_day38_report  # noqa: E402
from ai_journey.gpt_bigram import CorpusSource  # noqa: E402


def source_for(text: str) -> CorpusSource:
    payload = text.encode("utf-8")
    return CorpusSource(
        name="test corpus",
        url="https://example.test/corpus.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        license="test-only",
    )


class Day38CliTests(unittest.TestCase):
    text = "First Citizen:\nWe know the model.\n" * 20

    def arguments(self, root: Path) -> list[str]:
        return [
            "--corpus",
            str(root / "corpus.txt"),
            "--output",
            str(root / "day38.json"),
            "--checkpoint",
            str(root / "day38.pt"),
            "--steps",
            "4",
            "--batch-size",
            "4",
            "--block-size",
            "4",
            "--embedding-dim",
            "8",
            "--heads",
            "2",
            "--layers",
            "1",
            "--train-token-limit",
            "320",
            "--validation-token-limit",
            "80",
            "--validation-fraction",
            "0.2",
            "--sample-tokens",
            "8",
            "--sample-top-k",
            "5",
        ]

    def test_cli_writes_verified_report_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(self.text, encoding="utf-8")
            with mock.patch(
                "ai_journey.day38_cli.TINY_SHAKESPEARE", source_for(self.text)
            ), redirect_stdout(io.StringIO()) as stdout:
                result = main(self.arguments(root))
            report = json.loads((root / "day38.json").read_text(encoding="utf-8"))

            self.assertEqual(result, 0)
            self.assertTrue((root / "day38.pt").is_file())
            verify_day38_report(report)
            self.assertIn("steps=4", stdout.getvalue())
            self.assertIn("sample=", stdout.getvalue())

    def test_cli_rejects_output_that_aliases_the_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(self.text, encoding="utf-8")
            arguments = self.arguments(root)
            arguments[3] = str(corpus)
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(arguments)

            self.assertEqual(result, 2)
            self.assertIn("must be distinct", stderr.getvalue())
            self.assertEqual(corpus.read_text(encoding="utf-8"), self.text)
            self.assertFalse((root / "day38.pt").exists())

    def test_download_uses_the_pinned_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [*self.arguments(root), "--download"]
            source = source_for(self.text)
            with mock.patch(
                "ai_journey.day38_cli.TINY_SHAKESPEARE", source
            ), mock.patch(
                "ai_journey.day38_cli.fetch_verified_corpus",
                return_value=self.text,
            ) as fetch, redirect_stdout(io.StringIO()):
                result = main(arguments)

            self.assertEqual(result, 0)
            fetch.assert_called_once_with(root / "corpus.txt", source)


if __name__ == "__main__":
    unittest.main()
