import unittest

from ai_journey.bpe import ByteTokenizer, RegexTokenizer, TokenizerError
from ai_journey.token_encoder_stream import DocumentEncodingStream


class DocumentEncodingStreamTests(unittest.TestCase):
    def test_chunked_encoding_matches_one_shot_across_every_boundary(self):
        tokenizer = ByteTokenizer.train("banana bandana " * 20, vocab_size=280)
        payload = "banana🙂bandana banana".encode("utf-8")
        expected = tokenizer.encode_bytes(payload)
        for boundary in range(len(payload) + 1):
            stream = DocumentEncodingStream(tokenizer)
            self.assertEqual(stream.feed(payload[:boundary]), ())
            actual = stream.feed(payload[boundary:], final=True)
            self.assertEqual(actual, expected)

    def test_explicit_boundaries_prevent_cross_document_merges(self):
        tokenizer = ByteTokenizer(((97, 98),))
        stream = DocumentEncodingStream(tokenizer)
        first = stream.feed(b"a", document_end=True)
        second = stream.feed(b"b", final=True)
        self.assertEqual((*first, *second), (97, 98))
        self.assertEqual(tokenizer.encode_bytes(b"ab"), (256,))

    def test_rejects_boundary_aware_tokenizers_and_budget_overruns(self):
        with self.assertRaisesRegex(TypeError, "plain byte"):
            DocumentEncodingStream(RegexTokenizer())
        stream = DocumentEncodingStream(ByteTokenizer(), max_document_bytes=2)
        with self.assertRaisesRegex(TokenizerError, "document budget"):
            stream.feed(b"abc")

    def test_resumes_partial_binary_documents_with_exact_token_ids(self):
        tokenizer = ByteTokenizer(((0, 255), (256, 128)))
        stream = DocumentEncodingStream(tokenizer, max_document_bytes=32)
        self.assertEqual(stream.feed(b"\x00\xff"), ())
        resumed = DocumentEncodingStream(tokenizer, max_document_bytes=32)
        resumed.restore(stream.snapshot())
        actual = resumed.feed(b"\x80\xfe", final=True)
        self.assertEqual(actual, tokenizer.encode_bytes(b"\x00\xff\x80\xfe"))

    def test_restore_rejects_changed_tokenizers_and_malformed_buffers(self):
        stream = DocumentEncodingStream(ByteTokenizer(), max_document_bytes=8)
        stream.feed(b"abc")
        state = stream.snapshot()
        changed = DocumentEncodingStream(
            ByteTokenizer(((97, 98),)), max_document_bytes=8
        )
        with self.assertRaisesRegex(TokenizerError, "policy"):
            changed.restore(state)
        with self.assertRaisesRegex(TokenizerError, "buffer"):
            stream.restore({**state, "buffer": "***"})
        self.assertEqual(stream.snapshot(), state)


if __name__ == "__main__":
    unittest.main()
