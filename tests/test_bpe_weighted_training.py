from collections import Counter
import random
import unittest

from ai_journey.bpe import ByteTokenizer, merge_pair, _merge_weighted_streams
from ai_journey.tokenizer_training import (
    train_documents,
    BPETrainingSession,
    training_segments,
)


def reference_merges(documents, target, pattern=None):
    streams = [
        tuple(text.encode()) for text in training_segments(documents, pattern=pattern)
    ]
    merges = []
    while len(merges) + 256 < target:
        counts = Counter(pair for stream in streams for pair in zip(stream, stream[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda value: (-counts[value], value))
        if counts[pair] < 2:
            break
        streams = [merge_pair(stream, pair, 256 + len(merges)) for stream in streams]
        merges.append(pair)
    return tuple(merges)


class WeightedTrainingTests(unittest.TestCase):
    def test_duplicate_document_weights_preserve_every_canonical_merge(self):
        rng = random.Random(40)
        for _ in range(15):
            documents = tuple(
                rng.choice(("banana banana", "ababab", "नमस्ते hello")) for _ in range(20)
            )
            for pattern in (None, r"\w+|[^\w\s]+|\s+"):
                expected = reference_merges(documents, 280, pattern)
                self.assertEqual(
                    train_documents(documents, vocab_size=280, pattern=pattern).merges,
                    expected,
                )
                session = BPETrainingSession(documents, pattern=pattern)
                session.advance(7)
                restored = BPETrainingSession.restore(documents, session.snapshot())
                self.assertEqual(restored.advance(17).merges, expected)

    def test_converging_streams_add_weights_instead_of_overwriting(self):
        streams = Counter({(97, 98): 3, (256,): 5})
        self.assertEqual(
            _merge_weighted_streams(streams, (97, 98), 256), Counter({(256,): 8})
        )

    def test_overlap_multiplicities_match_unweighted_training(self):
        documents = ("aaaaa",) * 100
        self.assertEqual(
            train_documents(documents, vocab_size=280).merges,
            reference_merges(documents, 280),
        )
        self.assertEqual(
            ByteTokenizer.train("aaaaa" * 100, vocab_size=280).decode(
                ByteTokenizer.train("aaaaa" * 100, vocab_size=280).encode("aaaaa")
            ),
            "aaaaa",
        )
