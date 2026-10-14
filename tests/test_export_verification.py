from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.bpe import TokenizerError
from ai_journey.export_verification import verify_export
from ai_journey.parallel_corpus import Sentence
from ai_journey.tokenizer_sources import HINDI


class ExportVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = [root / name for name in ("manifest", "train", "validation")]
        parts = {}
        for name, identifier, text, path in (
            ("train", "a", "नमस्ते\n", self.paths[1]),
            ("validation", "b", "नई दुनिया\n", self.paths[2]),
        ):
            data = text.encode()
            path.write_bytes(data)
            parts[name] = {
                "sentence_ids": [identifier],
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        self.manifest = {
            "schema_version": 1,
            "source": asdict(HINDI),
            "source_url": HINDI.url,
            "license_url": HINDI.license_url,
            "attribution": "UD",
            "split_seed": 40,
            "unused_sentences": 0,
            "partitions": parts,
        }
        self.paths[0].write_text(json.dumps(self.manifest))

    def test_pinned_snapshot_proves_exported_ids_and_exact_text(self):
        records = (Sentence("a", "नमस्ते"), Sentence("b", "नई दुनिया"))
        with patch(
            "ai_journey.export_verification.load_public_corpus", return_value=records
        ):
            self.assertEqual(
                verify_export(*self.paths, source_path=Path("snapshot")), self.manifest
            )
        with patch(
            "ai_journey.export_verification.load_public_corpus",
            return_value=(records[0], Sentence("b", "changed")),
        ), self.assertRaisesRegex(TokenizerError, "pinned source"):
            verify_export(*self.paths, source_path=Path("snapshot"))

    def test_changed_partition_and_cross_partition_ids_fail(self):
        self.paths[1].write_text("tampered")
        with self.assertRaisesRegex(TokenizerError, "identity"):
            verify_export(*self.paths)
        self.paths[1].write_text("नमस्ते\n", encoding="utf-8")
        self.manifest["partitions"]["validation"]["sentence_ids"] = ["a"]
        self.paths[0].write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(TokenizerError, "disjoint"):
            verify_export(*self.paths)
