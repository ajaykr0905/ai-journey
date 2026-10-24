import unittest

from ai_journey.bpe import ByteTokenizer, SpecialTokenizer
from ai_journey.token_spans import encode_bytes_with_spans, encode_with_spans


class TokenSpanTests(unittest.TestCase):
    def test_arbitrary_byte_spans_cover_the_original_payload_exactly(self):
        tokenizer = ByteTokenizer(((0, 255), (256, 128)))
        payload = b"\x00\xff\x80\xfe"
        spans = encode_bytes_with_spans(tokenizer, payload)
        self.assertEqual(
            [(span.token_id, span.byte_start, span.byte_end) for span in spans],
            [(257, 0, 3), (254, 3, 4)],
        )
        rebuilt = b"".join(payload[span.byte_start : span.byte_end] for span in spans)
        self.assertEqual(rebuilt, payload)

    def test_partial_unicode_tokens_cover_one_character_without_losing_bytes(self):
        tokenizer = ByteTokenizer()
        text = "न😀x"
        spans = encode_with_spans(tokenizer, text)
        self.assertEqual(
            [(s.character_start, s.character_end) for s in spans[:3]], [(0, 1)] * 3
        )
        self.assertEqual(
            [(s.character_start, s.character_end) for s in spans[3:7]], [(1, 2)] * 4
        )
        for span in spans:
            self.assertEqual(
                text.encode()[span.byte_start : span.byte_end],
                tokenizer.token_bytes(span.token_id),
            )

    def test_merged_and_special_token_offsets_are_exact(self):
        tokenizer = SpecialTokenizer(ByteTokenizer.train("hello " * 30), ("<end>",))
        text = "hello <end>"
        spans = encode_with_spans(tokenizer, text, allowed=frozenset(tokenizer.names))
        self.assertEqual((spans[-1].byte_start, spans[-1].byte_end), (6, 11))
        self.assertEqual((spans[-1].character_start, spans[-1].character_end), (6, 11))
        self.assertEqual(encode_with_spans(tokenizer, ""), ())
