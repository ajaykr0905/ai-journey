import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ai_journey.transformer_certification import certify_transformer
from ai_journey.transformer_evidence import (
    build_certification_report,
    load_certification_report,
    verify_certification_report,
    write_certification_report,
)


class TransformerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.result = certify_transformer()
        self.report = build_certification_report(self.result)

    def test_atomic_evidence_roundtrip_and_owned_input(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "evidence.json")
            write_certification_report(path, self.report)
            self.assertEqual(load_certification_report(path), self.report)
        self.result["padding_error"] = 2
        verify_certification_report(self.report)

    def test_bad_measurements_are_rejected_even_with_recomputed_digest(self):
        for name, value in (
            ("padding_error", 1),
            ("cache_rollover_error", float("nan")),
            ("parameter_count", True),
        ):
            raw = copy.deepcopy(self.result)
            raw[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_certification_report(raw)
        raw = copy.deepcopy(self.result)
        raw["attention"][0]["causality"]["boundaries_checked"] = 0
        with self.assertRaisesRegex(ValueError, "boundaries"):
            build_certification_report(raw)

    def test_tampering_and_failed_publish_preserve_existing_bytes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "evidence.json")
            path.write_bytes(b"previous")
            report = copy.deepcopy(self.report)
            report["result"]["padding_error"] = 9
            with self.assertRaises(ValueError):
                write_certification_report(path, report)
            with patch(
                "ai_journey.transformer_evidence.os.replace",
                side_effect=OSError("injected"),
            ):
                with self.assertRaises(OSError):
                    write_certification_report(path, self.report)
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_duplicate_fields_are_rejected_before_hash_check(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "evidence.json")
            path.write_text('{"schema":"x","schema":"x"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_certification_report(path)
