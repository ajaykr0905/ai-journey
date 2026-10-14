"""BPE checkpoint restart must preserve weights, dropout, sampler, and codec."""

from pathlib import Path
import tempfile
import unittest

import torch

from ai_journey.bpe_corpus import build_bpe_corpus
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    load_training_checkpoint,
    model_fingerprint,
    save_training_checkpoint,
    seed_everything,
    train_steps,
)


class BPETransformerRestartTests(unittest.TestCase):
    def test_split_training_matches_continuous_dropout_training_exactly(self):
        corpus = build_bpe_corpus(
            ("hello hello world", "नमस्ते नमस्ते दुनिया"), ("held-out 🙂",), vocab_size=272
        )
        architecture = TransformerConfig(
            vocab_size=corpus.vocab_size,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            dropout=0.2,
        )
        config = TrainingConfig(steps=4, batch_size=2, seed=40)

        def initialize():
            seed_everything(40)
            model = DecoderLanguageModel(architecture)
            return (
                model,
                build_optimizer(model, config),
                BatchCursor(corpus.train_tokens, block_size=4, batch_size=2, seed=40),
            )

        continuous, optimizer, cursor = initialize()
        expected = train_steps(continuous, cursor, optimizer, config)
        expected_rng = torch.get_rng_state().clone()
        first, optimizer, cursor = initialize()
        before = train_steps(first, cursor, optimizer, config, step_count=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart.pt"
            save_training_checkpoint(
                path,
                model=first,
                optimizer=optimizer,
                cursor=cursor,
                training_config=config,
                corpus_fingerprint=corpus.fingerprint(),
                step=2,
            )
            resumed, optimizer, cursor = initialize()
            step = load_training_checkpoint(
                path,
                model=resumed,
                optimizer=optimizer,
                cursor=cursor,
                training_config=config,
                corpus_fingerprint=corpus.fingerprint(),
            )
            after = train_steps(
                resumed, cursor, optimizer, config, start_step=step, step_count=2
            )
        self.assertEqual(before + after, expected)
        self.assertEqual(model_fingerprint(resumed), model_fingerprint(continuous))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_same_sized_but_different_bpe_identity_cannot_restore_checkpoint(self):
        first = build_bpe_corpus(("abababab",), ("held out",), vocab_size=258)
        second = build_bpe_corpus(("cdcdcdcd",), ("held out",), vocab_size=258)
        self.assertEqual(first.vocab_size, second.vocab_size)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=first.vocab_size,
                block_size=2,
                embedding_dim=4,
                head_count=1,
                layer_count=1,
            )
        )
        config = TrainingConfig(steps=1, batch_size=1)
        optimizer = build_optimizer(model, config)
        cursor = BatchCursor(first.train_tokens, block_size=2, batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=config,
                corpus_fingerprint=first.fingerprint(),
                step=0,
            )
            with self.assertRaisesRegex(ValueError, "corpus fingerprint"):
                load_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    training_config=config,
                    corpus_fingerprint=second.fingerprint(),
                )
