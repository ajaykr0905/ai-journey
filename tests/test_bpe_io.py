import json
import tempfile
import unittest
from pathlib import Path

from ai_journey.bpe import (
    ByteTokenizer,
    RegexTokenizer,
    SpecialTokenizer,
    TokenizerError,
)
from ai_journey.tokenizer_io import (
    save_tokenizer,
    load_tokenizer,
    tokenizer_fingerprint,
)


class TokenizerCheckpointTests(unittest.TestCase):
    def test_all_variants_restart_with_identical_ids_and_fingerprints(self):
        text = "hello नमस्ते " * 20
        variants = [ByteTokenizer.train(text), RegexTokenizer.train(text)]
        variants.append(SpecialTokenizer(variants[1], ("<end>",)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            for original in variants:
                save_tokenizer(path, original)
                restored = load_tokenizer(path)
                self.assertEqual(restored.encode(text), original.encode(text))
                self.assertEqual(
                    tokenizer_fingerprint(restored), tokenizer_fingerprint(original)
                )

    def test_corruption_numeric_alias_and_duplicate_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(path, RegexTokenizer())
            payload = json.loads(path.read_text())
            payload["schema_version"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaises(TokenizerError):
                load_tokenizer(path)
            path.write_text('{"state":{},"state":{}}')
            with self.assertRaisesRegex(TokenizerError, "duplicate"):
                load_tokenizer(path)

    def test_regex_identity_cannot_drift_from_compiled_behavior(self):
        tokenizer = RegexTokenizer()
        with self.assertRaises(AttributeError):
            tokenizer.pattern = "changed"
