import unittest
from unittest.mock import patch

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.bpe_benchmark import benchmark_encoding


class EncoderBenchmarkTests(unittest.TestCase):
    def test_both_encoders_have_identical_hash_bound_token_sequences(self):
        documents = ("hello hello", "नमस्ते नमस्ते", "🙂 Unicode")
        tokenizer = ByteTokenizer.train("\n".join(documents), vocab_size=280)
        report = benchmark_encoding(tokenizer, documents, repeats=2)
        self.assertEqual(
            report["utf8_bytes"], sum(len(text.encode()) for text in documents)
        )
        self.assertEqual(
            report["tokens"], sum(len(tokenizer.encode(text)) for text in documents)
        )
        self.assertEqual(len(report["elapsed_ns"]["heap"]), 2)
        self.assertGreater(report["vocabulary_storage_bytes"], 256)

    def test_fast_encoder_regression_fails_instead_of_publishing_speed_claims(self):
        with patch(
            "ai_journey.bpe_benchmark.encode_fast", return_value=(0,)
        ), self.assertRaisesRegex(TokenizerError, "parity"):
            benchmark_encoding(ByteTokenizer(), ("public text",))
