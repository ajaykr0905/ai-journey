import unittest

from ai_journey.bpe import ByteTokenizer, TokenizerError
from ai_journey.token_stream import TokenTextStream


class TokenStreamTests(unittest.TestCase):
    def test_utf8_codepoints_resume_across_token_boundaries(self):
        tokenizer = ByteTokenizer()
        values = tokenizer.encode("न😀text")
        stream = TokenTextStream(tokenizer)
        self.assertEqual(stream.feed(values[:1]), "")
        restarted = TokenTextStream(tokenizer)
        restarted.restore(stream.snapshot())
        text = "".join(restarted.feed([value]) for value in values[1:])
        self.assertEqual(text + restarted.feed([], final=True), "न😀text")
        with self.assertRaises(TokenizerError):
            restarted.feed([97])

    def test_failed_decode_or_restore_does_not_replace_live_state(self):
        stream = TokenTextStream(ByteTokenizer())
        stream.feed([0xE0])
        before = stream.snapshot()
        with self.assertRaises(UnicodeError):
            stream.feed([255])
        self.assertEqual(before, stream.snapshot())
        with self.assertRaises(TokenizerError):
            stream.restore({**before, "buffer": "/w=="})
        self.assertEqual(before, stream.snapshot())
