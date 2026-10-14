from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from ai_journey.bpe import TokenizerError
from ai_journey.tokenizer_sources import HINDI, ENGLISH, load_public_corpus


class PublicSourceTests(unittest.TestCase):
    def test_pins_are_immutable_and_licenses_are_explicit(self):
        for source in (HINDI, ENGLISH):
            self.assertIn(source.revision, source.url)
            self.assertEqual(len(source.revision), 40)
            self.assertEqual(source.license, "CC-BY-SA-3.0")

    def test_changed_truncated_and_appended_snapshots_are_rejected(self):
        payload = b"# sent_id = one\n# text = public sentence\n"
        source = replace(
            ENGLISH, size_bytes=len(payload), sha256=sha256(payload).hexdigest()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus"
            path.write_bytes(payload)
            self.assertEqual(load_public_corpus(path, source)[0].sentence_id, "one")
            for changed in (
                payload[:-1],
                payload + b"x",
                payload.replace(b"public", b"secret"),
            ):
                path.write_bytes(changed)
                with self.assertRaises(TokenizerError):
                    load_public_corpus(path, source)
