import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_conformance import build_training_audit
from ai_journey.bpe_conformance_io import load_training_audit, save_training_audit
from ai_journey.tokenizer_training import train_documents


class TrainingAuditIOTests(unittest.TestCase):
    def setUp(self):
        self.documents = ("banana banana", "नमस्ते नमस्ते")
        tokenizer = train_documents(self.documents, vocab_size=270)
        self.report = build_training_audit(self.documents, tokenizer)

    def test_round_trips_audit_without_losing_merge_pairs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            save_training_audit(path, self.report)
            loaded = load_training_audit(path)
            self.assertEqual(loaded, json.loads(json.dumps(self.report)))

    def test_rejects_tampering_duplicate_fields_and_oversized_inputs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            path.write_text(json.dumps({**self.report, "final_tokens": 999}))
            with self.assertRaisesRegex(TokenizerError, "fingerprint"):
                load_training_audit(path)
            path.write_text('{"audit_sha256":"a","audit_sha256":"b"}')
            with self.assertRaisesRegex(TokenizerError, "duplicate"):
                load_training_audit(path)
            path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(TokenizerError, "2 MiB"):
                load_training_audit(path)


if __name__ == "__main__":
    unittest.main()
