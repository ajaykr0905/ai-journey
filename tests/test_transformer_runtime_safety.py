"""Rejected training inputs must not corrupt reproducible state."""

import copy
import random
import unittest
from unittest import mock

import numpy as np
import torch

from ai_journey.transformer_cache import generate_cached
from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TokenCorpus,
    CharacterCodec,
    TransformerConfig,
    build_optimizer,
    evaluate_nll,
    expected_parameter_count,
    seed_everything,
    train_steps,
)


class RuntimeSafetyTests(unittest.TestCase):
    def test_parameter_estimates_match_allocated_models(self):
        for expansion in (1, 4, 7):
            for normalization in ("layer_norm", "scratch_batch_norm"):
                config = TransformerConfig(
                    vocab_size=7,
                    block_size=5,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=3,
                    feed_forward_expansion=expansion,
                    normalization_mode=normalization,
                )
                self.assertEqual(
                    expected_parameter_count(config),
                    DecoderLanguageModel(config).parameter_count,
                )

    def setUp(self):
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )
        self.config = TrainingConfig(steps=2, batch_size=2)
        self.cursor = BatchCursor(torch.arange(30) % 5, block_size=4, batch_size=2)
        self.optimizer = build_optimizer(self.model, self.config)

    def test_invalid_seed_is_rejected_without_changing_any_rng(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seed_everything(seed)
            self.assertEqual(python_state, random.getstate())
            self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            with self.assertRaises(ValueError):
                TrainingConfig(seed=seed)

    def test_cpu_seeding_does_not_schedule_cuda_rng_mutations(self):
        with mock.patch("torch.cuda.manual_seed_all") as cuda_seed:
            seed_everything(38)
        cuda_seed.assert_not_called()

    def test_corpus_fingerprint_binds_partition_boundaries(self):
        codec = CharacterCodec(("a", "b"))
        first = TokenCorpus(
            codec, torch.tensor([0, 1, 0]), torch.tensor([1, 0]), "a" * 64
        )
        second = TokenCorpus(
            codec, torch.tensor([0, 1]), torch.tensor([0, 1, 0]), "a" * 64
        )
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_token_corpus_owns_validated_cpu_partitions(self):
        train = torch.tensor([0, 1, 0])
        corpus = TokenCorpus(CharacterCodec(("a", "b")), train, train, "a" * 64)
        fingerprint = corpus.fingerprint()
        train.fill_(1)
        self.assertEqual(fingerprint, corpus.fingerprint())
        for invalid in (
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1, 1]),
            torch.tensor([0, 2]),
            torch.tensor([[0, 1]]),
        ):
            with self.subTest(tokens=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                TokenCorpus(corpus.codec, invalid, corpus.validation_tokens, "a" * 64)

    def test_cursor_rejects_unreachable_or_unexecutable_restart_state(self):
        original = self.cursor.state_dict()
        for state in ({"epoch": 0, "offset": 1}, {"epoch": 2**64, "offset": 0}):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.cursor.load_state_dict(state)
            self.assertEqual(original, self.cursor.state_dict())
        with self.assertRaises(ValueError):
            BatchCursor(self.cursor.tokens, block_size=4, batch_size=2, seed=2**64)

    def test_nonfinite_training_never_reaches_the_optimizer(self):
        before = copy.deepcopy(self.model.state_dict())
        with mock.patch.object(self.optimizer, "step") as update:
            loss = torch.tensor(float("nan"), requires_grad=True)
            with mock.patch.object(self.model, "forward", return_value=(None, loss)):
                with self.assertRaisesRegex(ValueError, "finite"):
                    train_steps(
                        self.model,
                        self.cursor,
                        self.optimizer,
                        self.config,
                        step_count=1,
                    )
            update.assert_not_called()
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[name]))
        parameter = next(self.model.parameters())
        hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with self.assertRaisesRegex((ValueError, RuntimeError), "finite"):
                train_steps(
                    self.model, self.cursor, self.optimizer, self.config, step_count=1
                )
        finally:
            hook.remove()
        self.assertFalse(self.optimizer.state)

    def test_training_rejects_optimizer_for_another_model(self):
        other = copy.deepcopy(self.model)
        optimizer = build_optimizer(other, self.config)
        before = self.cursor.state_dict()
        with self.assertRaisesRegex(ValueError, "optimizer.*parameters"):
            train_steps(self.model, self.cursor, optimizer, self.config, step_count=1)
        self.assertEqual(before, self.cursor.state_dict())

    def test_training_rejects_incompatible_cursor_before_consuming_data(self):
        cursor = BatchCursor(self.cursor.tokens, block_size=5, batch_size=2)
        original = cursor.state_dict()
        with self.assertRaisesRegex(ValueError, "block_size"):
            train_steps(self.model, cursor, self.optimizer, self.config, step_count=1)
        self.assertEqual(original, cursor.state_dict())

    def test_evaluation_rejects_nonfinite_metrics_and_restores_modes(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        logits = torch.full((2, 4, 5), float("nan"))
        with mock.patch.object(self.model, "forward", return_value=(logits, None)):
            with self.assertRaisesRegex(ValueError, "finite"):
                evaluate_nll(self.model, self.cursor.tokens, batch_size=2)
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_evaluation_streams_weighted_losses_across_tail_batches(self):
        tokens = self.cursor.tokens[:17]
        expected = evaluate_nll(self.model, tokens, batch_size=64)
        with mock.patch(
            "torch.cat", side_effect=AssertionError("retained corpus losses")
        ):
            actual = evaluate_nll(self.model, tokens, batch_size=3)
        self.assertAlmostEqual(expected, actual, places=6)

    def test_greedy_generation_breaks_ties_with_argmax_for_both_paths(self):
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
        prompt = torch.tensor([[1, 2]])
        full = self.model.generate(
            prompt, new_tokens=8, top_k=1, generator=torch.Generator().manual_seed(37)
        )
        cached = generate_cached(self.model, prompt, new_tokens=8, top_k=1, seed=37)
        self.assertTrue(torch.equal(full[:, 2:], torch.zeros(1, 8, dtype=torch.long)))
        self.assertTrue(torch.equal(full, cached))

    def test_invalid_targets_do_not_update_batchnorm_statistics(self):
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                normalization_mode="scratch_batch_norm",
            )
        )
        before = copy.deepcopy(model.state_dict())
        with self.assertRaises(ValueError):
            model(
                torch.zeros(2, 4, dtype=torch.long),
                torch.full((2, 4), 5, dtype=torch.long),
            )
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]), name)
