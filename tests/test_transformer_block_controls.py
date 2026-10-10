from __future__ import annotations

import unittest

import torch

from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    FeedForward,
    TransformerConfig,
    TransformerBlock,
    TransformerLabError,
)


class GenerationInputGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=7, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )

    def test_zero_token_generation_validates_and_clones_full_prompt(self) -> None:
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        generated = self.model.generate(prompt, new_tokens=0)
        torch.testing.assert_close(generated, prompt)
        self.assertNotEqual(generated.data_ptr(), prompt.data_ptr())
        invalid = prompt.clone()
        invalid[0, 0] = 7
        with self.assertRaisesRegex(TransformerLabError, "vocabulary"):
            self.model.generate(invalid, new_tokens=0)

    def test_rejects_invalid_temperature_even_without_new_tokens(self) -> None:
        for temperature in (
            True,
            False,
            0,
            -1,
            float("nan"),
            float("inf"),
            -float("inf"),
            "hot",
        ):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                TransformerLabError, "temperature"
            ):
                self.model.generate(
                    torch.tensor([[1]]), new_tokens=0, temperature=temperature
                )

    def test_rejects_empty_malformed_and_out_of_vocabulary_prompts(self) -> None:
        for prompt in (
            torch.empty(1, 0, dtype=torch.long),
            torch.empty(0, 1, dtype=torch.long),
            torch.tensor([[-1]]),
            torch.tensor([[7]]),
        ):
            with self.subTest(prompt=prompt), self.assertRaises(TransformerLabError):
                self.model.generate(prompt, new_tokens=0)
        for prompt in (torch.tensor([[1.0]]), torch.tensor([1]), [[1]]):
            with self.assertRaises(TypeError):
                self.model.generate(prompt, new_tokens=0)

    def test_nonfinite_sampling_logits_fail_and_restore_mixed_module_modes(
        self,
    ) -> None:
        self.model.train()
        self.model.blocks[0].attention.eval()
        modes = [module.training for module in self.model.modules()]
        with torch.no_grad():
            self.model.lm_head.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(TransformerLabError, "sampling logits"):
            self.model.generate(torch.tensor([[1]]), new_tokens=1)
        self.assertEqual([module.training for module in self.model.modules()], modes)

    def test_seeded_generation_supports_long_prompts_and_preserves_inputs(self) -> None:
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        original = prompt.clone()
        first = self.model.generate(
            prompt, new_tokens=3, generator=torch.Generator().manual_seed(37)
        )
        second = self.model.generate(
            prompt, new_tokens=3, generator=torch.Generator().manual_seed(37)
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(prompt, original, rtol=0, atol=0)
        self.assertEqual(first.shape, (1, 8))


class NormalizationPlacementTests(unittest.TestCase):
    def test_block_implements_each_declared_residual_equation(self) -> None:
        for placement in ("pre", "post"):
            with self.subTest(placement=placement):
                config = TransformerConfig(
                    vocab_size=7,
                    embedding_dim=8,
                    head_count=2,
                    normalization_placement=placement,
                )
                block = TransformerBlock(config).double().eval()
                inputs = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
                actual = block(inputs)
                if placement == "pre":
                    residual = inputs + block.attention(block.attention_norm(inputs))
                    expected = residual + block.feed_forward(
                        block.feed_forward_norm(residual)
                    )
                else:
                    residual = block.attention_norm(inputs + block.attention(inputs))
                    expected = block.feed_forward_norm(
                        residual + block.feed_forward(residual)
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grad = torch.autograd.grad(
                    actual.square().sum(), inputs, retain_graph=True
                )[0]
                expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_legacy_defaults_preserve_config_fingerprint(self) -> None:
        import json
        from dataclasses import asdict
        from hashlib import sha256

        config = TransformerConfig(vocab_size=7)
        legacy = asdict(config)
        for name in (
            "normalization_placement",
            "feed_forward_expansion",
            "feed_forward_activation",
            "activation_checkpointing",
            "attention_backend",
        ):
            legacy.pop(name, None)
        expected = sha256(
            json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(config.fingerprint(), expected)
        self.assertNotEqual(
            config.fingerprint(),
            TransformerConfig(
                vocab_size=7, normalization_placement="post"
            ).fingerprint(),
        )

    def test_post_norm_keeps_padding_zero_and_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "normalization_placement"):
            TransformerConfig(vocab_size=7, normalization_placement="sandwich")
        block = TransformerBlock(
            TransformerConfig(
                vocab_size=7,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                normalization_placement="post",
            )
        )
        output = block(torch.randn(2, 4, 8), lengths=torch.tensor([1, 3]))
        self.assertEqual(torch.count_nonzero(output[0, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[1, 3:]).item(), 0)


class FeedForwardExpansionTests(unittest.TestCase):
    def test_expansion_controls_hidden_width_and_parameter_budget(self) -> None:
        for expansion in (1, 2, 4, 7):
            with self.subTest(expansion=expansion):
                config = TransformerConfig(
                    vocab_size=7,
                    embedding_dim=8,
                    head_count=2,
                    feed_forward_expansion=expansion,
                )
                mlp = FeedForward(config).double()
                self.assertEqual(mlp.network[0].out_features, 8 * expansion)
                self.assertEqual(mlp.network[2].in_features, 8 * expansion)
                self.assertEqual(
                    sum(p.numel() for p in mlp.parameters()),
                    2 * 8 * (8 * expansion) + 8 * expansion + 8,
                )
                inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
                output = mlp(inputs)
                self.assertEqual(output.shape, inputs.shape)
                output.square().sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in mlp.parameters()
                    )
                )

    def test_invalid_expansion_is_rejected_before_parameter_allocation(self) -> None:
        for value in (True, False, 1.5, "4"):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError, "feed_forward_expansion"
            ):
                TransformerConfig(vocab_size=7, feed_forward_expansion=value)
        for value in (0, -1):
            with self.assertRaisesRegex(TransformerLabError, "feed_forward_expansion"):
                TransformerConfig(vocab_size=7, feed_forward_expansion=value)

    def test_default_expansion_preserves_architecture_identity(self) -> None:
        default = TransformerConfig(vocab_size=7)
        self.assertEqual(
            default.fingerprint(),
            TransformerConfig(vocab_size=7, feed_forward_expansion=4).fingerprint(),
        )
        self.assertNotEqual(
            default.fingerprint(),
            TransformerConfig(vocab_size=7, feed_forward_expansion=2).fingerprint(),
        )


class FeedForwardActivationTests(unittest.TestCase):
    def test_activation_outputs_and_input_gradients_match_explicit_equations(
        self,
    ) -> None:
        for activation in ("gelu", "relu"):
            with self.subTest(activation=activation):
                config = TransformerConfig(
                    vocab_size=7,
                    embedding_dim=8,
                    head_count=2,
                    feed_forward_activation=activation,
                )
                mlp = FeedForward(config).double().eval()
                inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
                actual = mlp(inputs)
                projected = mlp.network[0](inputs)
                activated = (
                    torch.nn.functional.gelu(projected)
                    if activation == "gelu"
                    else torch.relu(projected)
                )
                expected = mlp.network[2](activated)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_grad = torch.autograd.grad(
                    actual.square().sum(), inputs, retain_graph=True
                )[0]
                expected_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_activation_policy_changes_identity_but_not_seeded_parameters(self) -> None:
        gelu = TransformerConfig(vocab_size=7, embedding_dim=8, head_count=2)
        relu = TransformerConfig(
            vocab_size=7, embedding_dim=8, head_count=2, feed_forward_activation="relu"
        )
        torch.manual_seed(37)
        first = DecoderLanguageModel(gelu)
        torch.manual_seed(37)
        second = DecoderLanguageModel(relu)
        self.assertNotEqual(gelu.fingerprint(), relu.fingerprint())
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIsInstance(first.blocks[0].feed_forward.network[1], torch.nn.GELU)
        self.assertIsInstance(second.blocks[0].feed_forward.network[1], torch.nn.ReLU)

    def test_rejects_unsupported_activation(self) -> None:
        for activation in ("tanh", "swish", "GELU", True):
            with self.subTest(activation=activation), self.assertRaisesRegex(
                TransformerLabError, "feed_forward_activation"
            ):
                TransformerConfig(vocab_size=7, feed_forward_activation=activation)


class ActivationCheckpointTests(unittest.TestCase):
    def test_checkpoint_training_preserves_dropout_rng_loss_and_all_gradients(
        self,
    ) -> None:
        from dataclasses import replace

        for placement in ("pre", "post"):
            with self.subTest(placement=placement):
                config = TransformerConfig(
                    vocab_size=11,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=2,
                    dropout=0.3,
                    normalization_placement=placement,
                )
                torch.manual_seed(37)
                eager = DecoderLanguageModel(config).double().train()
                checkpointed = (
                    DecoderLanguageModel(replace(config, activation_checkpointing=True))
                    .double()
                    .train()
                )
                checkpointed.load_state_dict(eager.state_dict())
                tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
                targets = torch.tensor([[2, 3, -100, -100], [6, 7, 8, 9]])
                lengths = torch.tensor([2, 4])
                torch.manual_seed(737)
                eager_logits, eager_loss = eager(tokens, targets, lengths=lengths)
                eager_loss.backward()
                eager_rng = torch.random.get_rng_state().clone()
                torch.manual_seed(737)
                actual_logits, actual_loss = checkpointed(
                    tokens, targets, lengths=lengths
                )
                actual_loss.backward()
                torch.testing.assert_close(actual_logits, eager_logits, rtol=0, atol=0)
                torch.testing.assert_close(actual_loss, eager_loss, rtol=0, atol=0)
                torch.testing.assert_close(
                    torch.random.get_rng_state(), eager_rng, rtol=0, atol=0
                )
                for actual, expected in zip(
                    checkpointed.parameters(), eager.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=0, atol=0
                    )

    def test_checkpointing_is_used_only_in_training_with_gradients(self) -> None:
        from unittest.mock import patch
        from ai_journey.transformer_lab import activation_checkpoint

        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=7,
                embedding_dim=8,
                head_count=2,
                layer_count=2,
                activation_checkpointing=True,
            )
        )
        tokens = torch.tensor([[1, 2, 3]])
        with patch(
            "ai_journey.transformer_lab.activation_checkpoint",
            wraps=activation_checkpoint,
        ) as checkpoint_call:
            model.train()
            model(tokens)
            self.assertEqual(checkpoint_call.call_count, 2)
            checkpoint_call.reset_mock()
            model.eval()
            model(tokens)
            model.train()
            with torch.no_grad():
                model(tokens)
            self.assertEqual(checkpoint_call.call_count, 0)

    def test_rejects_nonboolean_controls_and_stateful_normalization(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError, "activation_checkpointing"
            ):
                TransformerConfig(vocab_size=7, activation_checkpointing=value)
        with self.assertRaisesRegex(TransformerLabError, "stateless"):
            TransformerConfig(
                vocab_size=7,
                activation_checkpointing=True,
                normalization_mode="scratch_batch_norm",
            )


class LegacyCheckpointCompatibilityTests(unittest.TestCase):
    def _save_legacy_archive(self, path):
        from ai_journey.transformer_lab import (
            BatchCursor,
            TrainingConfig,
            build_optimizer,
            save_training_checkpoint,
            train_steps,
        )

        torch.manual_seed(37)
        config = TransformerConfig(
            vocab_size=7,
            block_size=3,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            dropout=0.2,
        )
        model = DecoderLanguageModel(config)
        training = TrainingConfig(steps=2, batch_size=2, seed=37)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(torch.arange(20) % 7, block_size=3, batch_size=2, seed=37)
        train_steps(model, cursor, optimizer, training, step_count=1)
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            cursor=cursor,
            training_config=training,
            corpus_fingerprint="a" * 64,
            step=1,
        )
        payload = torch.load(path, weights_only=True)
        for name in (
            "normalization_placement",
            "feed_forward_expansion",
            "feed_forward_activation",
            "activation_checkpointing",
            "attention_backend",
        ):
            payload["model_config"].pop(name)
        torch.save(payload, path)
        return model, optimizer, cursor, training

    def test_schema_one_legacy_defaults_restore_bit_exact_training_restart(
        self,
    ) -> None:
        import tempfile
        from pathlib import Path
        from ai_journey.transformer_lab import (
            BatchCursor,
            build_optimizer,
            load_training_checkpoint,
            train_steps,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model, optimizer, cursor, training = self._save_legacy_archive(path)
            expected = train_steps(
                model, cursor, optimizer, training, start_step=1, step_count=1
            )
            restored = DecoderLanguageModel(model.config)
            restored_optimizer = build_optimizer(restored, training)
            restored_cursor = BatchCursor(
                torch.arange(20) % 7, block_size=3, batch_size=2, seed=37
            )
            step = load_training_checkpoint(
                path,
                model=restored,
                optimizer=restored_optimizer,
                cursor=restored_cursor,
                training_config=training,
                corpus_fingerprint="a" * 64,
            )
            actual = train_steps(
                restored,
                restored_cursor,
                restored_optimizer,
                training,
                start_step=step,
                step_count=1,
            )
            self.assertEqual(actual, expected)
            self.assertEqual(restored_cursor.state_dict(), cursor.state_dict())
            for actual_parameter, expected_parameter in zip(
                restored.parameters(), model.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    actual_parameter, expected_parameter, rtol=0, atol=0
                )

    def test_legacy_defaults_cannot_override_nondefault_requested_model(self) -> None:
        import tempfile
        from dataclasses import replace
        from pathlib import Path
        from ai_journey.transformer_lab import build_optimizer, load_training_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model, _, cursor, training = self._save_legacy_archive(path)
            for change in (
                {"normalization_placement": "post"},
                {"feed_forward_expansion": 2},
                {"feed_forward_activation": "relu"},
                {"activation_checkpointing": True},
                {"attention_backend": "sdpa"},
            ):
                target = DecoderLanguageModel(replace(model.config, **change))
                optimizer = build_optimizer(target, training)
                states = {
                    name: tensor.clone() for name, tensor in target.state_dict().items()
                }
                cursor_state = cursor.state_dict().copy()
                rng = torch.random.get_rng_state().clone()
                with self.subTest(change=change), self.assertRaisesRegex(
                    TransformerLabError, "configuration mismatch"
                ):
                    load_training_checkpoint(
                        path,
                        model=target,
                        optimizer=optimizer,
                        cursor=cursor,
                        training_config=training,
                        corpus_fingerprint="a" * 64,
                    )
                for name, tensor in target.state_dict().items():
                    torch.testing.assert_close(tensor, states[name], rtol=0, atol=0)
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.assertEqual(cursor.state_dict(), cursor_state)
                torch.testing.assert_close(
                    torch.random.get_rng_state(), rng, rtol=0, atol=0
                )

    def test_unknown_controls_missing_original_fields_and_boolean_aliases_fail(
        self,
    ) -> None:
        import tempfile
        from pathlib import Path
        from ai_journey.transformer_lab import load_training_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            model, optimizer, cursor, training = self._save_legacy_archive(path)
            original = torch.load(path, weights_only=True)
            for change in ("unknown", "missing_original", "boolean_alias"):
                payload = dict(original)
                payload["model_config"] = dict(original["model_config"])
                if change == "unknown":
                    payload["model_config"]["future_backend"] = "unsafe"
                elif change == "missing_original":
                    payload["model_config"].pop("dropout")
                else:
                    payload["model_config"]["activation_checkpointing"] = 0
                torch.save(payload, path)
                with self.subTest(change=change), self.assertRaisesRegex(
                    TransformerLabError, "configuration"
                ):
                    load_training_checkpoint(
                        path,
                        model=model,
                        optimizer=optimizer,
                        cursor=cursor,
                        training_config=training,
                        corpus_fingerprint="a" * 64,
                    )
