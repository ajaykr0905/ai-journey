from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import (
    BatchCursor,
    CharacterCodec,
    TokenCorpus,
    TransformerConfig,
    TransformerLabError,
)


class TransformerConfigTests(unittest.TestCase):
    def test_config_validates_dimensions_and_has_stable_fingerprint(self) -> None:
        config = TransformerConfig(vocab_size=27, embedding_dim=24, head_count=3)
        self.assertEqual(config.head_dim, 8)
        self.assertEqual(config.fingerprint(), config.fingerprint())
        self.assertEqual(len(config.fingerprint()), 64)

    def test_config_rejects_incompatible_heads(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "divisible"):
            TransformerConfig(vocab_size=27, embedding_dim=10, head_count=3)


class CharacterCodecTests(unittest.TestCase):
    def test_codec_round_trips_public_text(self) -> None:
        codec = CharacterCodec.from_text("anna\naria\n")
        encoded = codec.encode("aria\n")
        self.assertEqual(codec.decode(list(encoded)), "aria\n")
        self.assertEqual(codec.tokens, tuple(sorted(set("anna\naria\n"))))

    def test_codec_rejects_unknown_characters(self) -> None:
        codec = CharacterCodec.from_text("ab")
        with self.assertRaisesRegex(TransformerLabError, "unknown"):
            codec.encode("abc")


class TokenCorpusTests(unittest.TestCase):
    def test_corpus_split_is_disjoint_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("anna\naria\namara\n" * 6, encoding="utf-8")
            corpus = TokenCorpus.from_path(path, block_size=4)
        self.assertGreater(len(corpus.train_tokens), len(corpus.validation_tokens))
        self.assertEqual(
            corpus.codec.decode(
                [*corpus.train_tokens.tolist(), *corpus.validation_tokens.tolist()]
            ),
            "anna\naria\namara\n" * 6,
        )
        self.assertEqual(len(corpus.fingerprint()), 64)


class BatchCursorTests(unittest.TestCase):
    def test_cursor_visits_each_window_once_per_epoch(self) -> None:
        import torch

        cursor = BatchCursor(torch.arange(8), block_size=3, batch_size=2, seed=7)
        starts: list[int] = []
        for _ in range(3):
            x, y = cursor.next()
            starts.extend(x[:, 0].tolist())
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertEqual(sorted(starts), list(range(5)))

    def test_cursor_state_restores_the_next_batch(self) -> None:
        import torch

        first = BatchCursor(torch.arange(12), block_size=3, batch_size=2, seed=4)
        first.next()
        state = first.state_dict()
        expected = first.next()
        resumed = BatchCursor(torch.arange(12), block_size=3, batch_size=2, seed=4)
        resumed.load_state_dict(state)
        actual = resumed.next()
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))


if __name__ == "__main__":
    unittest.main()
