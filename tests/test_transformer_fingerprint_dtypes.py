"""Fingerprint dtype support must preserve existing checkpoint hashes."""

from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    build_optimizer,
    load_training_checkpoint,
    model_fingerprint,
    save_training_checkpoint,
    train_steps,
)


class TransformerFingerprintDtypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_rng = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, self.original_rng)
        self.config = TransformerConfig(
            vocab_size=8,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            dropout=0.1,
        )

    def test_bfloat16_hash_is_repeatable_and_detects_parameter_mutation(self) -> None:
        model = DecoderLanguageModel(self.config).to(dtype=torch.bfloat16)
        fingerprint = model_fingerprint(model)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(model_fingerprint(model), fingerprint)
        with torch.no_grad():
            next(model.parameters()).reshape(-1)[0] += 1
        self.assertNotEqual(model_fingerprint(model), fingerprint)

    def test_numpy_compatible_dtypes_keep_legacy_hashes(self) -> None:
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = DecoderLanguageModel(self.config).to(dtype=dtype)
                # Include scalars, empty tensors and noncontiguous buffers.
                model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
                model.register_buffer("empty", torch.empty(0, dtype=dtype))
                model.register_buffer("flags", torch.tensor([True, False]))
                model.register_buffer("indices", torch.arange(6, dtype=torch.int32))
                model.register_buffer("transpose", torch.arange(6.0).reshape(2, 3).T)
                legacy = sha256()
                for name, tensor in sorted(model.state_dict().items()):
                    value = tensor.detach().cpu().contiguous()
                    legacy.update(name.encode())
                    legacy.update(str(value.dtype).encode())
                    legacy.update(str(tuple(value.shape)).encode())
                    legacy.update(value.numpy().tobytes())
                self.assertEqual(model_fingerprint(model), legacy.hexdigest())

    def test_bfloat16_checkpoint_resume_restores_cpu_training_state(self) -> None:
        torch.random.default_generator.manual_seed(41)
        training = TrainingConfig(steps=2, batch_size=2, seed=41)
        tokens = torch.arange(40) % self.config.vocab_size
        model = DecoderLanguageModel(self.config).to(dtype=torch.bfloat16)
        optimizer = build_optimizer(model, training)
        cursor = BatchCursor(tokens, block_size=4, batch_size=2, seed=training.seed)
        train_steps(model, cursor, optimizer, training, step_count=1)
        saved_fingerprint = model_fingerprint(model)
        saved_cursor = cursor.state_dict()
        saved_rng = torch.get_rng_state().clone()
        saved_optimizer = optimizer.state_dict()
        self.assertTrue(saved_optimizer["state"])
        for state in saved_optimizer["state"].values():
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bfloat16.pt"
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
            self.assertEqual(payload["model_fingerprint"], saved_fingerprint)
            self.assertTrue(
                all(
                    value.dtype == torch.bfloat16
                    for value in payload["model_state"].values()
                )
            )
            expected_trace = train_steps(
                model, cursor, optimizer, training, start_step=1, step_count=1
            )
            expected_fingerprint = model_fingerprint(model)
            expected_rng = torch.get_rng_state().clone()

            resumed = DecoderLanguageModel(self.config).to(dtype=torch.bfloat16)
            resumed_optimizer = build_optimizer(resumed, training)
            resumed_cursor = BatchCursor(
                tokens, block_size=4, batch_size=2, seed=training.seed
            )
            start_step = load_training_checkpoint(
                path,
                model=resumed,
                optimizer=resumed_optimizer,
                cursor=resumed_cursor,
                training_config=training,
                corpus_fingerprint="a" * 64,
            )
            self.assertEqual(start_step, 1)
            self.assertEqual(model_fingerprint(resumed), saved_fingerprint)
            self.assertEqual(resumed_cursor.state_dict(), saved_cursor)
            self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng))
            restored_optimizer = resumed_optimizer.state_dict()
            self.assertEqual(
                restored_optimizer["param_groups"],
                payload["optimizer_state"]["param_groups"],
            )
            for index, state in payload["optimizer_state"]["state"].items():
                for name, value in state.items():
                    self.assertTrue(
                        torch.equal(restored_optimizer["state"][index][name], value),
                        (index, name),
                    )
            actual_trace = train_steps(
                resumed,
                resumed_cursor,
                resumed_optimizer,
                training,
                start_step=start_step,
                step_count=1,
            )
        self.assertEqual(actual_trace, expected_trace)
        self.assertEqual(model_fingerprint(resumed), expected_fingerprint)
        self.assertEqual(resumed_cursor.state_dict(), cursor.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))


if __name__ == "__main__":
    unittest.main()
