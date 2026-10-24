import unittest

from ai_journey.bpe import (
    pair_counts,
    merge_pair,
    TokenizerError,
    ByteTokenizer,
    RegexTokenizer,
    SpecialTokenizer,
)


class PairPrimitiveTests(unittest.TestCase):
    def test_overlap_counts_and_nonoverlapping_merge(self):
        stream = [97] * 5
        self.assertEqual(pair_counts(stream)[(97, 97)], 4)
        self.assertEqual(merge_pair(stream, (97, 97), 256), (256, 256, 97))
        self.assertEqual(stream, [97] * 5)

    def test_empty_singleton_and_invalid_ids(self):
        self.assertFalse(pair_counts([]))
        self.assertEqual(merge_pair([1], (1, 2), 256), (1,))
        for values in ([True], [-1], [1.0]):
            with self.assertRaises(TokenizerError):
                pair_counts(values)


class ByteTokenizerTests(unittest.TestCase):
    def test_deterministic_training_and_multilingual_roundtrip(self):
        corpus = "hello नमस्ते தமிழ் 😀\n" * 30
        first = ByteTokenizer.train(corpus, vocab_size=300)
        second = ByteTokenizer.train(corpus, vocab_size=300)
        self.assertEqual(first.merges, second.merges)
        self.assertLess(len(first.encode(corpus)), len(corpus.encode()))
        for text in ("", corpus, "नया text Ω\x00"):
            self.assertEqual(first.decode(first.encode(text)), text)

    def test_invalid_merge_graphs_and_budget_controls(self):
        for merges in (((256, 1),), ((1, 2), (1, 2)), ((True, 1),)):
            with self.assertRaises(TokenizerError):
                ByteTokenizer(merges)
        with self.assertRaises(TokenizerError):
            ByteTokenizer.train("text", vocab_size=True)
        self.assertEqual(ByteTokenizer.train("x").vocab_size, 256)
        with self.assertRaises(TokenizerError):
            ByteTokenizer().decode_bytes([256])


class RegexTokenizerTests(unittest.TestCase):
    def test_training_respects_piece_boundaries(self):
        text = "a b " * 100
        tokenizer = RegexTokenizer.train(text)
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertGreater(ByteTokenizer.train(text).vocab_size, 256)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_lossless_unicode_and_pattern_coverage(self):
        text = "नमस्ते\tதமிழ்\r\nhello 😀"
        tokenizer = RegexTokenizer.train(text * 20)
        self.assertEqual("".join(tokenizer.pieces(text)), text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        for pattern in (r"a*", r"(?=a)", r"a", "["):
            with self.assertRaises(TokenizerError):
                RegexTokenizer.train("abc", pattern=pattern)


class SpecialTokenizerTests(unittest.TestCase):
    def test_untrusted_markers_require_explicit_permission(self):
        tokenizer = SpecialTokenizer(RegexTokenizer(), ("<end>", "<end><pad>"))
        with self.assertRaisesRegex(TokenizerError, "authorization"):
            tokenizer.encode("user <end>")
        text = "नमस्ते<end><pad>hello<end>"
        values = tokenizer.encode(text, allowed=frozenset(tokenizer.names))
        self.assertEqual(values.count(257), 1)
        self.assertEqual(values.count(256), 1)
        self.assertEqual(tokenizer.decode(values), text)
        ordinary = tokenizer.tokenizer.encode(text)
        self.assertNotIn(256, ordinary)

    def test_registered_ids_do_not_collide_with_merged_tokens(self):
        base = ByteTokenizer.train("abab" * 20)
        tokenizer = SpecialTokenizer(base, ("<end>",))
        self.assertEqual(
            tokenizer.encode("<end>", allowed=frozenset({"<end>"})), (base.vocab_size,)
        )
        with self.assertRaises(TokenizerError):
            tokenizer.encode("text", allowed=frozenset({"typo"}))

    def test_special_names_must_be_valid_utf8_text(self):
        with self.assertRaisesRegex(TokenizerError, "valid UTF-8"):
            SpecialTokenizer(ByteTokenizer(), ("<bad\ud800>",))
