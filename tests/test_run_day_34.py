from __future__ import annotations

import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_journey import day34_cli as run_day_34
from ai_journey.gpt_bigram import (
    CorpusSource,
    load_checkpoint,
    verify_experiment_report,
)


def fixture_source(payload: bytes) -> CorpusSource:
    return CorpusSource(
        "fixture",
        "https://example.test/fixture.txt",
        sha256(payload).hexdigest(),
        len(payload),
        "CC0-1.0",
    )


class Day34CLITests(unittest.TestCase):
    def test_cli_publishes_report_and_complete_checkpoint(self) -> None:
        payload = (("abcabc\n" * 60) + ("cab\n" * 20)).encode()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input.txt"
            output = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            corpus.write_bytes(payload)
            with patch.object(run_day_34, "TINY_SHAKESPEARE", fixture_source(payload)):
                code = run_day_34.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--output",
                        str(output),
                        "--checkpoint",
                        str(checkpoint),
                        "--steps",
                        "5",
                        "--batch-size",
                        "4",
                        "--block-size",
                        "3",
                        "--sample-tokens",
                        "8",
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(output.read_text())
            verify_experiment_report(report)
            self.assertEqual(report["completed_step"], 5)
            self.assertEqual(load_checkpoint(checkpoint)["step"], 5)

    def test_invalid_corpus_preserves_existing_outputs(self) -> None:
        payload = b"abcabcabc"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input.txt"
            output = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            corpus.write_bytes(b"wrong")
            output.write_text("preserve-report")
            checkpoint.write_bytes(b"preserve-checkpoint")
            with patch.object(run_day_34, "TINY_SHAKESPEARE", fixture_source(payload)):
                code = run_day_34.main(
                    [
                        "--corpus",
                        str(corpus),
                        "--output",
                        str(output),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(), "preserve-report")
            self.assertEqual(checkpoint.read_bytes(), b"preserve-checkpoint")


if __name__ == "__main__":
    unittest.main()
