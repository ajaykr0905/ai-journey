from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_model_ablation import VocabularyProtocol
from ai_journey.bpe_model_cli import evidence_fingerprint, load_protocol, main
from ai_journey.tokenizer_sources import PublicCorpus


class BPEModelCLITests(unittest.TestCase):
    def test_executes_pinned_fixture_pipeline_and_keeps_aligned_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, sources = [], []
            for language in ("hi", "en"):
                data = "".join(
                    f"# sent_id = s{index}\n# text = {language} banana banana abracadabra sentence {index}\n\n"
                    for index in range(6)
                ).encode()
                path = root / f"{language}.conllu"
                path.write_bytes(data)
                paths.append(path)
                sources.append(
                    PublicCorpus(
                        language,
                        "public-test/fixture",
                        "a" * 40,
                        path.name,
                        sha256(data).hexdigest(),
                        len(data),
                    )
                )
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps(
                    asdict(
                        VocabularyProtocol(
                            vocab_sizes=(256, 258, 260),
                            seeds=(1, 2),
                            steps=1,
                            block_size=3,
                        )
                    )
                )
            )
            output = root / "comparison.json"
            argv = [
                "--hindi",
                str(paths[0]),
                "--english",
                str(paths[1]),
                "--protocol",
                str(protocol),
                "--output",
                str(output),
                "--train-count",
                "2",
                "--validation-count",
                "2",
            ]
            with patch("ai_journey.bpe_model_cli.HINDI", sources[0]), patch(
                "ai_journey.bpe_model_cli.ENGLISH", sources[1]
            ), redirect_stdout(StringIO()):
                self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["sha256"], evidence_fingerprint(report))
            self.assertEqual(len(report["languages"]), 2)
            left, right = report["languages"]
            self.assertEqual(left["train_sentence_ids"], right["train_sentence_ids"])
            self.assertEqual(
                left["validation_sentence_ids"], right["validation_sentence_ids"]
            )
            self.assertFalse(
                set(left["train_sentence_ids"]) & set(left["validation_sentence_ids"])
            )
            self.assertEqual(left["unused_sentences"], 2)
            self.assertEqual(len(left["experiment"]["trials"]), 6)
            before = paths[0].read_bytes()
            with redirect_stderr(StringIO()):
                argv[7] = str(paths[0])
                self.assertEqual(main(argv), 2)
            self.assertEqual(paths[0].read_bytes(), before)

    def test_duplicate_or_oversized_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for text in ('{"steps":1,"steps":2}', " " * 16385):
                path.write_text(text)
                with self.assertRaises(TokenizerError):
                    load_protocol(path)
