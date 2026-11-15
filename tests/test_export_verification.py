from contextlib import redirect_stdout
from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_journey.bpe import TokenizerError
from ai_journey.export_verification import verify_export
from ai_journey.parallel_corpus import Sentence
from ai_journey.parallel_export import main as export_main
from ai_journey.tokenizer_sources import ENGLISH, HINDI


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

    def test_manifest_must_declare_split_seed_40(self):
        self.manifest["split_seed"] = 41
        self.paths[0].write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(TokenizerError, "split controls"):
            verify_export(*self.paths)


def ranked_ids(count):
    return sorted(
        (f"s{index}" for index in range(count)),
        key=lambda value: (sha256(f"40:{value}".encode()).digest(), value),
    )


class SeedMembershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = [root / name for name in ("manifest", "train", "validation")]
        self.ranked = ranked_ids(30)
        self.records = tuple(
            Sentence(value, f"sentence {value}") for value in sorted(self.ranked)
        )

    def write_export(self, train, validation):
        parts = {}
        for name, ids, path in (
            ("train", train, self.paths[1]),
            ("validation", validation, self.paths[2]),
        ):
            data = "".join(f"sentence {value}\n" for value in ids).encode()
            path.write_bytes(data)
            parts[name] = {
                "sentence_ids": list(ids),
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        manifest = {
            "schema_version": 1,
            "source": asdict(HINDI),
            "source_url": HINDI.url,
            "license_url": HINDI.license_url,
            "attribution": "UD",
            "split_seed": 40,
            "unused_sentences": len(self.records) - len(train) - len(validation),
            "partitions": parts,
        }
        self.paths[0].write_text(json.dumps(manifest))
        return manifest

    def verify_with_snapshot(self):
        with patch(
            "ai_journey.export_verification.load_public_corpus",
            return_value=self.records,
        ):
            return verify_export(*self.paths, source_path=Path("snapshot"))

    def test_seed_ranked_export_verifies_with_and_without_snapshot(self):
        manifest = self.write_export(self.ranked[:10], self.ranked[10:20])
        self.assertEqual(verify_export(*self.paths), manifest)
        self.assertEqual(self.verify_with_snapshot(), manifest)

    def test_swapped_train_and_heldout_sentence_is_rejected(self):
        train = [self.ranked[10], *self.ranked[1:10]]
        validation = [self.ranked[0], *self.ranked[11:20]]
        self.write_export(train, validation)
        with self.assertRaisesRegex(TokenizerError, "train"):
            verify_export(*self.paths)
        with self.assertRaisesRegex(TokenizerError, "train"):
            self.verify_with_snapshot()

    def test_heldout_that_skips_a_ranked_id_is_rejected_with_snapshot(self):
        manifest = self.write_export(self.ranked[:10], self.ranked[11:21])
        self.assertEqual(verify_export(*self.paths), manifest)
        with self.assertRaisesRegex(TokenizerError, "validation"):
            self.verify_with_snapshot()

    def test_train_that_is_not_the_ranked_prefix_is_rejected_with_snapshot(self):
        self.write_export(self.ranked[1:11], self.ranked[11:21])
        with self.assertRaisesRegex(TokenizerError, "train"):
            self.verify_with_snapshot()

    def test_reordered_partition_is_rejected_without_snapshot(self):
        validation = [self.ranked[11], self.ranked[10], *self.ranked[12:20]]
        self.write_export(self.ranked[:10], validation)
        with self.assertRaisesRegex(TokenizerError, "validation"):
            verify_export(*self.paths)


class ExporterRoundTripTests(unittest.TestCase):
    def test_default_exports_verify_with_and_without_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for source, word in ((HINDI, "नमस्ते"), (ENGLISH, "hello")):
                data = "\n\n".join(
                    f"# sent_id = n{index}\n# text = {word} {index}"
                    for index in range(30)
                ).encode()
                (root / source.language).write_bytes(data)
                sources[source.language] = replace(
                    source, sha256=sha256(data).hexdigest(), size_bytes=len(data)
                )
            for language in ("hi", "en"):
                paths = [root / f"{language}.{name}" for name in ("json", "tr", "va")]
                with (
                    patch("ai_journey.parallel_export.HINDI", sources["hi"]),
                    patch("ai_journey.parallel_export.ENGLISH", sources["en"]),
                    redirect_stdout(io.StringIO()),
                ):
                    export_main(
                        [
                            "--hindi",
                            str(root / "hi"),
                            "--english",
                            str(root / "en"),
                            "--language",
                            language,
                            "--train-output",
                            str(paths[1]),
                            "--validation-output",
                            str(paths[2]),
                            "--manifest",
                            str(paths[0]),
                        ]
                    )
                with (
                    self.subTest(language=language),
                    patch("ai_journey.export_verification.HINDI", sources["hi"]),
                    patch("ai_journey.export_verification.ENGLISH", sources["en"]),
                ):
                    manifest = verify_export(*paths)
                    self.assertEqual(
                        verify_export(*paths, source_path=root / language), manifest
                    )
                    self.assertEqual(
                        len(manifest["partitions"]["train"]["sentence_ids"]), 10
                    )
                    self.assertEqual(manifest["unused_sentences"], 10)
