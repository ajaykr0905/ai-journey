from collections import Counter
import random
import unittest

from ai_journey.bpe import (
    pair_counts,
    merge_pair,
    TokenizerError,
    ByteTokenizer,
    RegexTokenizer,
    SpecialTokenizer,
    _merge_weighted_streams,
    _weighted_pair_counts,
)


def scan_merge(values, pair, replacement):
    """Reference merge that inspects one position at a time, left to right."""
    output = []
    index = 0
    while index < len(values):
        if tuple(values[index : index + 2]) == pair:
            output.append(replacement)
            index += 2
        else:
            output.append(values[index])
            index += 1
    return tuple(output)


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

    def test_merge_matches_one_position_scan_on_random_and_edge_streams(self):
        rng = random.Random(71)
        streams = [(), (1,), (1, 2), (2, 1), (1, 1, 1), (1, 1, 1, 1), (3, 1)]
        streams += [(1, 3, 1), (2, 1, 2, 1, 1)]
        streams += [
            tuple(rng.choice((1, 1, 2, 3)) for _ in range(rng.randrange(40)))
            for _ in range(300)
        ]
        for stream in streams:
            for pair in ((1, 1), (1, 2), (2, 1), (1, 3), (1, 4)):
                self.assertEqual(
                    merge_pair(stream, pair, 256), scan_merge(stream, pair, 256)
                )

    def test_weighted_helpers_match_explicitly_expanded_streams(self):
        rng = random.Random(72)
        for _ in range(100):
            streams = Counter(
                {
                    tuple(rng.choice((1, 2, 3)) for _ in range(rng.randrange(12))): (
                        rng.randint(1, 4)
                    )
                    for _ in range(rng.randint(1, 8))
                }
            )
            pair = (rng.choice((1, 2, 3)), rng.choice((1, 2, 3)))
            counts = Counter()
            merged = Counter()
            for stream, weight in streams.items():
                for _ in range(weight):
                    counts.update(zip(stream, stream[1:]))
                    merged[scan_merge(stream, pair, 256)] += 1
            self.assertEqual(dict(_weighted_pair_counts(streams)), dict(counts))
            self.assertEqual(
                dict(_merge_weighted_streams(streams, pair, 256)), dict(merged)
            )


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

    def test_raw_byte_encoding_round_trips_non_utf8_payloads(self):
        tokenizer = ByteTokenizer(((0, 255), (256, 128)))
        payload = b"\x00\xff\x80\xfe"
        ids = tokenizer.encode_bytes(payload)
        self.assertEqual(ids, (257, 254))
        self.assertEqual(tokenizer.decode_bytes(ids), payload)
        text = "नमस्ते🙂"
        self.assertEqual(
            tokenizer.encode_bytes(text.encode("utf-8")), tokenizer.encode(text)
        )

    def test_encoding_applies_the_lowest_ranked_present_pair_first(self):
        rng = random.Random(73)
        for _ in range(20):
            text = "".join(rng.choice("aab नम🙂 ") for _ in range(60))
            tokenizer = ByteTokenizer.train(text * 2, vocab_size=290)
            ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
            for sample in ("", "a", text, text[::-1]):
                values = tuple(sample.encode("utf-8"))
                while True:
                    present = [
                        pair for pair in zip(values, values[1:]) if pair in ranks
                    ]
                    if not present:
                        break
                    pair = min(present, key=ranks.__getitem__)
                    values = scan_merge(values, pair, 256 + ranks[pair])
                self.assertEqual(tokenizer.encode(sample), values)


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
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            RegexTokenizer.train(123)
        with self.assertRaisesRegex(TokenizerError, "decoded text"):
            tokenizer.encode_bytes(b"hello")


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

    def test_markers_match_leftmost_first_then_longest(self):
        tokenizer = SpecialTokenizer(ByteTokenizer(), ("ab", "bcd"))
        for allowed in (frozenset(tokenizer.names), frozenset({"ab"})):
            # The earlier "ab" beats the longer "bcd", whose rest stays ordinary.
            self.assertEqual(tokenizer.encode("abcd", allowed=allowed), (256, 99, 100))
        with self.assertRaisesRegex(TokenizerError, "authorization"):
            tokenizer.encode("bcd", allowed=frozenset({"ab"}))

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
