import random
import unittest

from ai_journey.bpe import ByteTokenizer, RegexTokenizer
from ai_journey.bpe_fast import encode_fast


class HeapEncodingTests(unittest.TestCase):
    def test_random_multilingual_and_overlapping_pairs_match_reference_exactly(self):
        rng = random.Random(40)
        for _ in range(30):
            text = "".join(rng.choice("aaabbc नमस्ते!🙂\n") for _ in range(80))
            tokenizer = ByteTokenizer.train(text * 3, vocab_size=280)
            for sample in ("", "aaaaaa", text, text[::-1], "未知 Unicode"):
                self.assertEqual(
                    encode_fast(tokenizer, sample), tokenizer.encode(sample)
                )

    def test_regex_boundaries_and_utf8_fallback_match_reference(self):
        tokenizer = RegexTokenizer.train("hello hello नमस्ते नमस्ते", vocab_size=280)
        for text in ("hello नमस्ते", "hello!hello", "🙂क्ष", "\n  "):
            self.assertEqual(encode_fast(tokenizer, text), tokenizer.encode(text))
