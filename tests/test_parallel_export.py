from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.parallel_export import main
from ai_journey.tokenizer_sources import HINDI, ENGLISH


class ParallelExportTests(unittest.TestCase):
    def test_export_preserves_text_and_disjoint_ids_with_hash_bound_license_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for source, word in ((HINDI, "नमस्ते"), (ENGLISH, "hello")):
                data = "\n\n".join(
                    f"# sent_id = {index}\n# text = {word} {index}"
                    for index in range(10)
                ).encode()
                (root / source.language).write_bytes(data)
                sources.append(
                    replace(
                        source, sha256=sha256(data).hexdigest(), size_bytes=len(data)
                    )
                )
            args = [
                "--hindi",
                str(root / "hi"),
                "--english",
                str(root / "en"),
                "--train-output",
                str(root / "train"),
                "--validation-output",
                str(root / "validation"),
                "--manifest",
                str(root / "manifest"),
                "--train-count",
                "3",
                "--validation-count",
                "2",
            ]
            with patch("ai_journey.parallel_export.HINDI", sources[0]), patch(
                "ai_journey.parallel_export.ENGLISH", sources[1]
            ), redirect_stdout(io.StringIO()):
                main(args)
            manifest = json.loads((root / "manifest").read_text())
            self.assertEqual(manifest["source"]["license"], "CC-BY-SA-3.0")
            self.assertEqual(manifest["unused_sentences"], 5)
            partitions = manifest["partitions"]
            self.assertFalse(
                set(partitions["train"]["sentence_ids"])
                & set(partitions["validation"]["sentence_ids"])
            )
            for name in partitions:
                self.assertEqual(
                    partitions[name]["sha256"],
                    sha256((root / name).read_bytes()).hexdigest(),
                )

    def test_aliases_are_rejected_before_any_source_is_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "input"
            corpus.write_text("preserve")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(
                    [
                        "--hindi",
                        str(corpus),
                        "--english",
                        str(root / "other"),
                        "--train-output",
                        str(corpus),
                        "--validation-output",
                        str(root / "validation"),
                        "--manifest",
                        str(root / "manifest"),
                    ]
                )
            self.assertEqual(corpus.read_text(), "preserve")
