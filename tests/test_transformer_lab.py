from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.transformer_lab import (
    BatchCursor,
    CausalSelfAttention,
    CharacterCodec,
    DecoderLanguageModel,
    FeedForward,
    TokenCorpus,
    TrainingConfig,
    TransformerBlock,
    TransformerConfig,
    TransformerLabError,
    build_normalization,
    build_optimizer,
    evaluate_nll,
    expected_initialization_std,
    load_training_checkpoint,
    model_fingerprint,
    run_transformer_experiment,
    save_training_checkpoint,
    seed_everything,
    train_steps,
    write_experiment_report,
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

    def test_config_rejects_invalid_initialization_scale(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "initialization_std"):
            TransformerConfig(vocab_size=27, initialization_std=0)

    def test_config_rejects_invalid_initialization_policy(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "initialization_mode"):
            TransformerConfig(vocab_size=27, initialization_mode="xavier")
        for gain in (0, float("inf"), float("nan"), True):
            with (
                self.subTest(gain=gain),
                self.assertRaisesRegex(TransformerLabError, "initialization_gain"),
            ):
                TransformerConfig(vocab_size=27, initialization_gain=gain)

    def test_seed_everything_repeats_model_initialization(self) -> None:
        import torch

        config = TransformerConfig(vocab_size=5, embedding_dim=8, head_count=2)
        seed_everything(41)
        first = DecoderLanguageModel(config)
        seed_everything(41)
        second = DecoderLanguageModel(config)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first.parameters(), second.parameters(), strict=True
                )
            )
        )

    def test_initialization_scale_changes_parameter_distribution(self) -> None:
        base = TransformerConfig(
            vocab_size=7, embedding_dim=32, head_count=4, initialization_std=0.01
        )
        stressed = TransformerConfig(
            vocab_size=7, embedding_dim=32, head_count=4, initialization_std=0.5
        )
        seed_everything(25)
        base_model = DecoderLanguageModel(base)
        seed_everything(25)
        stressed_model = DecoderLanguageModel(stressed)
        base_std = float(base_model.token_embedding.weight.detach().std())
        stressed_std = float(stressed_model.token_embedding.weight.detach().std())
        self.assertGreater(stressed_std, base_std * 20)
        self.assertNotEqual(base.fingerprint(), stressed.fingerprint())

    def test_kaiming_std_uses_linear_fan_in_and_explicit_gain(self) -> None:
        import math

        from torch import nn

        config = TransformerConfig(
            vocab_size=7,
            initialization_mode="kaiming_normal",
            initialization_gain=1.5,
        )
        self.assertEqual(
            expected_initialization_std(nn.Linear(9, 4), config),
            1.5 / math.sqrt(9),
        )
        self.assertEqual(
            expected_initialization_std(nn.Embedding(9, 4), config),
            config.initialization_std,
        )
        with self.assertRaisesRegex(TypeError, "Linear or torch.nn.Embedding"):
            expected_initialization_std(nn.LayerNorm(4), config)

    def test_kaiming_policy_changes_linear_but_not_embedding_initialization(
        self,
    ) -> None:
        import torch

        fixed = TransformerConfig(
            vocab_size=31,
            embedding_dim=64,
            head_count=4,
            layer_count=1,
            initialization_std=0.02,
        )
        kaiming = TransformerConfig(
            vocab_size=31,
            embedding_dim=64,
            head_count=4,
            layer_count=1,
            initialization_std=0.02,
            initialization_mode="kaiming_normal",
            initialization_gain=1.0,
        )
        seed_everything(26)
        fixed_model = DecoderLanguageModel(fixed)
        seed_everything(26)
        kaiming_model = DecoderLanguageModel(kaiming)
        self.assertTrue(
            torch.equal(
                fixed_model.token_embedding.weight,
                kaiming_model.token_embedding.weight,
            )
        )
        fixed_weight = fixed_model.blocks[0].feed_forward.network[0].weight
        kaiming_weight = kaiming_model.blocks[0].feed_forward.network[0].weight
        self.assertGreater(
            float(kaiming_weight.detach().std()),
            float(fixed_weight.detach().std()) * 4,
        )

    def test_training_config_rejects_non_positive_controls(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "steps"):
            TrainingConfig(steps=0)
        with self.assertRaisesRegex(TransformerLabError, "learning_rate"):
            TrainingConfig(learning_rate=0)

    def test_normalization_policy_builds_validated_layers(self) -> None:
        from torch import nn

        from ai_journey.batch_normalization import ScratchBatchNorm

        layer = build_normalization(
            TransformerConfig(vocab_size=7, embedding_dim=12, head_count=3)
        )
        self.assertIsInstance(layer, nn.LayerNorm)
        config = TransformerConfig(
            vocab_size=7,
            embedding_dim=12,
            head_count=3,
            normalization_mode="scratch_batch_norm",
            batch_norm_eps=2e-5,
            batch_norm_momentum=0.25,
        )
        scratch = build_normalization(config)
        self.assertIsInstance(scratch, ScratchBatchNorm)
        self.assertEqual(scratch.eps, 2e-5)
        self.assertEqual(scratch.momentum, 0.25)
        with self.assertRaisesRegex(TransformerLabError, "normalization_mode"):
            TransformerConfig(vocab_size=7, normalization_mode="group_norm")
        for value in (0, float("inf"), True):
            with self.subTest(value=value), self.assertRaises(TransformerLabError):
                TransformerConfig(vocab_size=7, batch_norm_eps=value)
        for value in (0, 1.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(TransformerLabError):
                TransformerConfig(vocab_size=7, batch_norm_momentum=value)


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


class CausalSelfAttentionTests(unittest.TestCase):
    def test_future_inputs_do_not_change_past_outputs(self) -> None:
        import torch

        torch.manual_seed(3)
        attention = CausalSelfAttention(
            TransformerConfig(
                vocab_size=8, block_size=5, embedding_dim=12, head_count=3
            )
        ).eval()
        original = torch.randn(2, 5, 12)
        changed = original.clone()
        changed[:, 3:] += 100
        with torch.no_grad():
            before = attention(original)
            after = attention(changed)
        self.assertTrue(torch.equal(before[:, :3], after[:, :3]))


class FeedForwardTests(unittest.TestCase):
    def test_feed_forward_preserves_token_shape_and_backpropagates(self) -> None:
        import torch

        layer = FeedForward(TransformerConfig(vocab_size=8, embedding_dim=12)).eval()
        inputs = torch.randn(2, 4, 12, requires_grad=True)
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        outputs.square().mean().backward()
        self.assertIsNotNone(inputs.grad)


class TransformerBlockTests(unittest.TestCase):
    def test_block_preserves_residual_shape_and_parameter_gradients(self) -> None:
        import torch

        block = TransformerBlock(
            TransformerConfig(
                vocab_size=8, block_size=6, embedding_dim=12, head_count=3
            )
        )
        inputs = torch.randn(2, 6, 12, requires_grad=True)
        block(inputs).mean().backward()
        self.assertEqual(inputs.grad.shape, inputs.shape)
        self.assertTrue(
            all(parameter.grad is not None for parameter in block.parameters())
        )

    def test_scratch_batchnorm_policy_updates_both_block_statistics(self) -> None:
        import torch

        from ai_journey.batch_normalization import ScratchBatchNorm

        block = TransformerBlock(
            TransformerConfig(
                vocab_size=8,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                normalization_mode="scratch_batch_norm",
            )
        )
        outputs = block(torch.randn(3, 4, 8))
        self.assertEqual(outputs.shape, (3, 4, 8))
        layers = [
            module for module in block.modules() if isinstance(module, ScratchBatchNorm)
        ]
        self.assertEqual(len(layers), 2)
        self.assertEqual([int(layer.num_batches_tracked) for layer in layers], [1, 1])


class DecoderLanguageModelTests(unittest.TestCase):
    def test_scratch_batchnorm_policy_covers_blocks_and_final_stream(self) -> None:
        import torch

        from ai_journey.batch_normalization import ScratchBatchNorm

        config = TransformerConfig(
            vocab_size=9,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=2,
            normalization_mode="scratch_batch_norm",
        )
        model = DecoderLanguageModel(config)
        token_ids = torch.randint(0, config.vocab_size, (3, config.block_size))
        logits, loss = model(token_ids, token_ids)
        self.assertEqual(logits.shape, (3, config.block_size, config.vocab_size))
        self.assertIsNotNone(loss)
        layers = [
            module for module in model.modules() if isinstance(module, ScratchBatchNorm)
        ]
        self.assertEqual(len(layers), 2 * config.layer_count + 1)
        self.assertTrue(all(int(layer.num_batches_tracked) == 1 for layer in layers))

    def test_model_returns_token_logits_and_cross_entropy(self) -> None:
        import torch

        config = TransformerConfig(
            vocab_size=7, block_size=5, embedding_dim=12, head_count=3, layer_count=2
        )
        model = DecoderLanguageModel(config)
        tokens = torch.randint(0, config.vocab_size, (3, config.block_size))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (3, 5, 7))
        self.assertIsNotNone(loss)
        self.assertGreater(model.parameter_count, 0)

    def test_model_rejects_out_of_vocabulary_tokens(self) -> None:
        import torch

        model = DecoderLanguageModel(TransformerConfig(vocab_size=3))
        with self.assertRaisesRegex(TransformerLabError, "vocabulary"):
            model(torch.tensor([[0, 3]]))

    def test_seeded_generation_is_repeatable_beyond_the_context_window(self) -> None:
        import torch

        torch.manual_seed(2)
        model = DecoderLanguageModel(
            TransformerConfig(vocab_size=5, block_size=3, embedding_dim=8, head_count=2)
        )
        prompt = torch.tensor([[0, 1, 2]])
        first = model.generate(
            prompt,
            new_tokens=5,
            top_k=3,
            generator=torch.Generator().manual_seed(9),
        )
        second = model.generate(
            prompt,
            new_tokens=5,
            top_k=3,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (1, 8))

    def test_optimizer_excludes_biases_and_norms_from_weight_decay(self) -> None:
        model = DecoderLanguageModel(
            TransformerConfig(vocab_size=5, embedding_dim=8, head_count=2)
        )
        optimizer = build_optimizer(model, TrainingConfig(weight_decay=0.2))
        groups = {
            group["weight_decay"]: group["params"] for group in optimizer.param_groups
        }
        self.assertEqual(set(groups), {0.0, 0.2})
        self.assertTrue(all(parameter.ndim >= 2 for parameter in groups[0.2]))
        self.assertTrue(all(parameter.ndim < 2 for parameter in groups[0.0]))

    def test_evaluation_is_finite_and_restores_training_mode(self) -> None:
        import math

        import torch

        model = DecoderLanguageModel(
            TransformerConfig(vocab_size=5, block_size=4, embedding_dim=8, head_count=2)
        ).train()
        loss = evaluate_nll(model, torch.arange(30) % 5, batch_size=3)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(model.training)

    def test_training_emits_finite_step_and_gradient_metrics(self) -> None:
        import math

        import torch

        seed_everything(11)
        model = DecoderLanguageModel(
            TransformerConfig(vocab_size=5, block_size=4, embedding_dim=8, head_count=2)
        )
        training = TrainingConfig(steps=3, batch_size=4, learning_rate=0.01)
        cursor = BatchCursor(
            torch.arange(40) % 5,
            block_size=model.config.block_size,
            batch_size=training.batch_size,
            seed=training.seed,
        )
        metrics = train_steps(model, cursor, build_optimizer(model, training), training)
        self.assertEqual([metric.step for metric in metrics], [1, 2, 3])
        self.assertTrue(all(math.isfinite(metric.loss) for metric in metrics))
        self.assertTrue(all(math.isfinite(metric.gradient_norm) for metric in metrics))

    def test_checkpoint_restores_model_optimizer_cursor_and_rng(self) -> None:
        import torch

        seed_everything(8)
        model_config = TransformerConfig(
            vocab_size=5, block_size=4, embedding_dim=8, head_count=2
        )
        training = TrainingConfig(steps=2, batch_size=3)
        model = DecoderLanguageModel(model_config)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(
            torch.arange(30) % 5, block_size=4, batch_size=3, seed=training.seed
        )
        train_steps(model, cursor, optimizer, training, step_count=1)
        expected_fingerprint = model_fingerprint(model)
        expected_cursor = cursor.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                corpus_fingerprint="a" * 64,
                step=1,
            )
            with torch.no_grad():
                next(model.parameters()).add_(1)
            cursor.next()
            step = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                corpus_fingerprint="a" * 64,
            )
            self.assertFalse(path.with_name(".checkpoint.pt.tmp").exists())
        self.assertEqual(step, 1)
        self.assertEqual(model_fingerprint(model), expected_fingerprint)
        self.assertEqual(cursor.state_dict(), expected_cursor)

    def test_checkpoint_rejects_a_different_initialization_policy(self) -> None:
        import torch

        training = TrainingConfig(steps=1, batch_size=3)
        kaiming_config = TransformerConfig(
            vocab_size=5,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            initialization_mode="kaiming_normal",
            initialization_gain=1.0,
        )
        seed_everything(26)
        model = DecoderLanguageModel(kaiming_config)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(
            torch.arange(30) % 5,
            block_size=4,
            batch_size=3,
            seed=training.seed,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kaiming.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                corpus_fingerprint="c" * 64,
                step=0,
            )
            fixed_model = DecoderLanguageModel(
                TransformerConfig(
                    vocab_size=5,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    initialization_gain=1.0,
                )
            )
            with self.assertRaisesRegex(
                TransformerLabError, "model configuration mismatch"
            ):
                load_training_checkpoint(
                    path,
                    model=fixed_model,
                    optimizer=build_optimizer(fixed_model, training),
                    cursor=BatchCursor(
                        torch.arange(30) % 5,
                        block_size=4,
                        batch_size=3,
                        seed=training.seed,
                    ),
                    training_config=training,
                    corpus_fingerprint="c" * 64,
                )

    def test_checkpoint_restart_matches_uninterrupted_training_bit_exactly(
        self,
    ) -> None:
        import torch

        seed_everything(12)
        model_config = TransformerConfig(
            vocab_size=5,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            dropout=0.1,
        )
        training = TrainingConfig(steps=2, batch_size=3, learning_rate=0.005)
        tokens = torch.arange(40) % 5
        model = DecoderLanguageModel(model_config)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(tokens, block_size=4, batch_size=3, seed=training.seed)
        train_steps(model, cursor, optimizer, training, step_count=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                training_config=training,
                corpus_fingerprint="b" * 64,
                step=1,
            )
            expected_metric = train_steps(
                model, cursor, optimizer, training, start_step=1, step_count=1
            )[0]
            expected_fingerprint = model_fingerprint(model)

            resumed_model = DecoderLanguageModel(model_config)
            resumed_optimizer = build_optimizer(resumed_model, training)
            resumed_cursor = BatchCursor(
                tokens, block_size=4, batch_size=3, seed=training.seed
            )
            start_step = load_training_checkpoint(
                path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                cursor=resumed_cursor,
                training_config=training,
                corpus_fingerprint="b" * 64,
            )
            actual_metric = train_steps(
                resumed_model,
                resumed_cursor,
                resumed_optimizer,
                training,
                start_step=start_step,
                step_count=1,
            )[0]
        self.assertEqual(actual_metric, expected_metric)
        self.assertEqual(model_fingerprint(resumed_model), expected_fingerprint)


class TransformerExperimentTests(unittest.TestCase):
    def test_experiment_writes_reproducible_training_evidence(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            corpus_path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
            corpus = TokenCorpus.from_path(corpus_path, block_size=4)
            result = run_transformer_experiment(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(
                    steps=4, batch_size=4, learning_rate=0.02
                ),
                checkpoint_path=root / "checkpoint.pt",
            )
            report_path = root / "report.json"
            write_experiment_report(report_path, result)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["completed_steps"], 4)
        self.assertEqual(len(payload["trace"]), 4)
        self.assertEqual(len(payload["model_fingerprint"]), 64)
        self.assertTrue(result.final_train_nll < result.initial_train_nll)


if __name__ == "__main__":
    unittest.main()
