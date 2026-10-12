from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.day38_training import (  # noqa: E402
    Day38DataConfig,
    prepare_bounded_corpus,
)
from ai_journey.gpt_bigram import CorpusSource, GPTBigramError  # noqa: E402
from ai_journey.transformer_lab import TransformerLabError  # noqa: E402


def source_for(text: str) -> CorpusSource:
    payload = text.encode("utf-8")
    return CorpusSource(
        name="test corpus",
        url="https://example.test/corpus.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        license="test-only",
    )


class Day38DataTests(unittest.TestCase):
    def test_bounded_partitions_are_disjoint_and_ordered(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz" * 8
        config = Day38DataConfig(
            block_size=4,
            train_token_limit=40,
            validation_token_limit=12,
            validation_fraction=0.25,
        )

        prepared = prepare_bounded_corpus(text, source_for(text), config)

        encoded = prepared.corpus.codec.encode(text)
        self.assertEqual(prepared.split_index, 156)
        self.assertEqual(prepared.full_train_token_count, 156)
        self.assertEqual(prepared.full_validation_token_count, 52)
        self.assertEqual(prepared.used_train_token_count, 40)
        self.assertEqual(prepared.used_validation_token_count, 12)
        self.assertEqual(prepared.corpus.train_tokens.tolist(), list(encoded[:40]))
        self.assertEqual(
            prepared.corpus.validation_tokens.tolist(), list(encoded[156:168])
        )

    def test_source_digest_is_verified_before_tokenization(self) -> None:
        text = "abcdefghij" * 8
        source = source_for(text)

        with self.assertRaises(GPTBigramError):
            prepare_bounded_corpus(
                text + "x",
                source,
                Day38DataConfig(
                    block_size=4,
                    train_token_limit=40,
                    validation_token_limit=12,
                ),
            )

    def test_budgets_must_exceed_context_window(self) -> None:
        with self.assertRaises(TransformerLabError):
            Day38DataConfig(
                block_size=8,
                train_token_limit=8,
                validation_token_limit=9,
            )

    def test_small_source_split_is_rejected(self) -> None:
        text = "abcdefghij"
        with self.assertRaises(TransformerLabError):
            prepare_bounded_corpus(
                text,
                source_for(text),
                Day38DataConfig(
                    block_size=4,
                    train_token_limit=8,
                    validation_token_limit=8,
                    validation_fraction=0.1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
