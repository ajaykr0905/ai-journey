from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.day40_cli import main
from ai_journey.tokenizer_sources import HINDI, ENGLISH


class Day40CliTests(unittest.TestCase):
    def test_verified_parallel_snapshots_publish_provenance_and_six_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            paths = []
            for source, word in ((HINDI, "नमस्ते"), (ENGLISH, "hello")):
                payload = "\n\n".join(
                    f"# sent_id = {index}\n# text = {word} {word} {index}"
                    for index in range(5)
                ).encode()
                path = root / source.language
                path.write_bytes(payload)
                paths.append(str(path))
                sources.append(
                    replace(
                        source,
                        sha256=sha256(payload).hexdigest(),
                        size_bytes=len(payload),
                    )
                )
            output = root / "report"
            with patch("ai_journey.day40_cli.HINDI", sources[0]), patch(
                "ai_journey.day40_cli.ENGLISH", sources[1]
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--hindi",
                            paths[0],
                            "--english",
                            paths[1],
                            "--output",
                            str(output),
                            "--train-count",
                            "3",
                        ]
                    ),
                    0,
                )
            report = json.loads(output.read_text())
            self.assertEqual(len(report["evaluations"]), 6)
            self.assertEqual(report["sources"][0]["sha256"], sources[0].sha256)

    def test_corrupt_corpora_do_not_publish_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.write_text("changed data")
            output = root / "report"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(
                    [
                        "--hindi",
                        str(corpus),
                        "--english",
                        str(corpus),
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())
