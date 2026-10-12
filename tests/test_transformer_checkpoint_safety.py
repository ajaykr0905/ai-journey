"""Checkpoint publication must preserve the previous recoverable state."""

from __future__ import annotations

import os
import copy
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from ai_journey.transformer_lab import (
    BatchCursor,
    DecoderLanguageModel,
    TrainingConfig,
    TransformerConfig,
    load_training_checkpoint,
    model_fingerprint,
    save_training_checkpoint,
)


class CheckpointPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_rng = torch.get_rng_state().clone()
        self.training = TrainingConfig(steps=3, batch_size=2)
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=5, block_size=4, embedding_dim=8, head_count=2, layer_count=1
            )
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.cursor = BatchCursor(torch.arange(30) % 5, block_size=4, batch_size=2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(torch.set_rng_state, self.original_rng)
        self.path = Path(self.directory.name) / "checkpoint.pt"

    def save(self, step: int = 1) -> None:
        save_training_checkpoint(
            self.path,
            model=self.model,
            optimizer=self.optimizer,
            cursor=self.cursor,
            training_config=self.training,
            corpus_fingerprint="a" * 64,
            step=step,
        )

    def test_success_syncs_serialized_state_before_replacement(self) -> None:
        with patch("os.fsync", wraps=os.fsync) as sync:
            self.save()
        self.assertGreaterEqual(sync.call_count, 1)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            ),
            1,
        )

    def test_checkpoint_rejects_unusable_adamw_controls(self) -> None:
        group = self.optimizer.param_groups[0]
        for name, value in (
            ("lr", -1.0),
            ("eps", 0.0),
            ("betas", (2.0, 0.9)),
            ("weight_decay", -1.0),
            ("amsgrad", "yes"),
        ):
            original = group[name]
            group[name] = value
            try:
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "AdamW"
                ):
                    self.save()
                self.assertFalse(self.path.exists())
            finally:
                group[name] = original

    def test_serialization_failure_preserves_checkpoint_and_removes_partial_file(self):
        self.save()
        original = self.path.read_bytes()

        def fail_mid_write(payload, stream):
            if isinstance(stream, Path):
                stream.write_bytes(b"partial archive")
            else:
                stream.write(b"partial archive")
            raise OSError("simulated serialization failure")

        with patch("torch.save", side_effect=fail_mid_write):
            with self.assertRaisesRegex(OSError, "serialization failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_sync_failure_keeps_previous_checkpoint(self) -> None:
        self.save()
        original = self.path.read_bytes()
        with patch("os.fsync", side_effect=OSError("simulated disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_directory_sync_failure_reports_uncertain_new_checkpoint(self) -> None:
        self.save()
        real_fsync = os.fsync
        sync_count = 0

        def fail_directory_sync(descriptor: int) -> None:
            nonlocal sync_count
            sync_count += 1
            if sync_count == 2:
                raise OSError("simulated directory sync failure")
            real_fsync(descriptor)

        with patch("os.fsync", side_effect=fail_directory_sync):
            with self.assertRaisesRegex(OSError, "directory sync failure"):
                self.save(2)
        self.assertEqual(sync_count, 2)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            ),
            2,
        )

    def test_replacement_failure_keeps_previous_checkpoint(self) -> None:
        self.save()
        original = self.path.read_bytes()
        with patch("os.replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaisesRegex(OSError, "replacement failure"):
                self.save(2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_corpus_fingerprint_must_be_a_sha256_digest(self) -> None:
        for fingerprint in ("z" * 64, "A" * 64, "a" * 63, b"a" * 64):
            with self.subTest(fingerprint=fingerprint), self.assertRaisesRegex(
                ValueError, "SHA-256"
            ):
                save_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint=fingerprint,
                    step=1,
                )

        self.save()
        payload = torch.load(self.path, weights_only=True)
        payload["corpus_fingerprint"] = "z" * 64
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="z" * 64,
            )

    def test_restore_rejects_malformed_checkpoint_containers(self) -> None:
        self.save()
        valid = torch.load(self.path, weights_only=True)
        malformed = [
            [valid],
            {key: value for key, value in valid.items() if key != "cursor_state"},
            {**valid, "unexpected": True},
        ]
        for payload in malformed:
            torch.save(payload, self.path)
            with self.subTest(payload_type=type(payload)), self.assertRaisesRegex(
                ValueError, "checkpoint fields"
            ):
                load_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )

    def test_restore_rejects_numeric_type_aliases_in_checkpoint_metadata(self) -> None:
        self.save()
        valid = torch.load(self.path, weights_only=True)
        malformed = []
        boolean_schema = copy.deepcopy(valid)
        boolean_schema["schema_version"] = True
        malformed.append(("boolean schema", boolean_schema))
        floating_steps = copy.deepcopy(valid)
        floating_steps["training_config"]["steps"] = float(self.training.steps)
        malformed.append(("floating steps", floating_steps))

        for defect, payload in malformed:
            torch.save(payload, self.path)
            with self.subTest(defect=defect), self.assertRaisesRegex(
                ValueError, "checkpoint"
            ):
                load_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )

    def test_malformed_model_fingerprint_is_rejected_before_model_restore(self) -> None:
        self.save()
        payload = torch.load(self.path, weights_only=True)
        payload["model_fingerprint"] = "z" * 64
        torch.save(payload, self.path)

        with patch.object(
            self.model, "load_state_dict", wraps=self.model.load_state_dict
        ) as restore:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )
        restore.assert_not_called()

    def test_concurrent_writers_publish_complete_independent_archives(self) -> None:
        barrier = threading.Barrier(2, timeout=10)
        real_save = torch.save

        def synchronize_writes(payload, stream):
            real_save(payload, stream)
            barrier.wait()

        with patch("torch.save", side_effect=synchronize_writes):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.save, step) for step in (1, 2)]
                for future in futures:
                    future.result(timeout=15)
        payload = torch.load(self.path, weights_only=True)
        self.assertIn(payload["step"], (1, 2))
        self.assertEqual(payload["model_fingerprint"], model_fingerprint(self.model))
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def populate_optimizer(self) -> None:
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.optimizer.step()

    def test_nonfinite_model_state_cannot_replace_a_valid_checkpoint(self) -> None:
        self.save()
        original = self.path.read_bytes()
        for value in (float("nan"), float("inf"), -float("inf")):
            parameter = next(self.model.parameters())
            before = parameter.detach().clone()
            with torch.no_grad():
                parameter.view(-1)[0] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                self.save(2)
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(list(self.path.parent.iterdir()), [self.path])
            with torch.no_grad():
                parameter.copy_(before)

    def test_nonfinite_optimizer_state_cannot_replace_a_valid_checkpoint(self) -> None:
        self.populate_optimizer()
        self.save()
        original = self.path.read_bytes()
        parameter = next(self.model.parameters())
        for field in ("exp_avg", "exp_avg_sq", "step"):
            tensor = self.optimizer.state[parameter][field]
            before = tensor.clone()
            tensor.view(-1)[0] = float("nan")
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                self.save(2)
            self.assertEqual(self.path.read_bytes(), original)
            tensor.copy_(before)
        self.optimizer.param_groups[0]["lr"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.save(2)
        self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_optimizer_moments_are_rejected_before_resume(self) -> None:
        self.populate_optimizer()
        self.save()
        valid = torch.load(self.path, weights_only=True)
        model_before = model_fingerprint(self.model)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        cursor_before = self.cursor.state_dict()
        rng_before = torch.get_rng_state().clone()
        for field in ("exp_avg", "exp_avg_sq", "step"):
            payload = copy.deepcopy(valid)
            state = next(iter(payload["optimizer_state"]["state"].values()))
            state[field].view(-1)[0] = float("inf")
            torch.save(payload, self.path)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                load_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )
            self.assertEqual(model_fingerprint(self.model), model_before)
            self.assertEqual(self.cursor.state_dict(), cursor_before)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            for key, expected in optimizer_before["state"].items():
                for name, tensor in expected.items():
                    self.assertTrue(
                        torch.equal(
                            self.optimizer.state_dict()["state"][key][name], tensor
                        )
                    )

    def test_nonfinite_model_with_matching_hash_is_rejected_before_resume(self) -> None:
        self.save()
        payload = torch.load(self.path, weights_only=True)
        corrupt = copy.deepcopy(self.model)
        with torch.no_grad():
            next(corrupt.parameters()).view(-1)[0] = float("nan")
        payload["model_state"] = corrupt.state_dict()
        payload["model_fingerprint"] = model_fingerprint(corrupt)
        torch.save(payload, self.path)
        before = model_fingerprint(self.model)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            )
        self.assertEqual(model_fingerprint(self.model), before)

    def test_optimizer_cast_overflow_rolls_back_the_restore(self) -> None:
        self.populate_optimizer()
        self.save()
        payload = torch.load(self.path, weights_only=True)
        state = next(iter(payload["optimizer_state"]["state"].values()))
        state["exp_avg"] = torch.full_like(state["exp_avg"], 1e100, dtype=torch.float64)
        torch.save(payload, self.path)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        with self.assertRaisesRegex(ValueError, "finite"):
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            )
        for key, expected in optimizer_before["state"].items():
            for name, tensor in expected.items():
                self.assertTrue(
                    torch.equal(self.optimizer.state_dict()["state"][key][name], tensor)
                )

    def test_malformed_adamw_state_is_rejected_before_the_next_training_step(self):
        self.populate_optimizer()
        self.save()
        valid = torch.load(self.path, weights_only=True)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        for defect in ("moment_shape", "missing_moment", "negative_variance", "step"):
            payload = copy.deepcopy(valid)
            state = next(iter(payload["optimizer_state"]["state"].values()))
            if defect == "moment_shape":
                state["exp_avg"] = torch.zeros(1)
            elif defect == "missing_moment":
                del state["exp_avg_sq"]
            elif defect == "negative_variance":
                state["exp_avg_sq"].view(-1)[0] = -1
            else:
                state["step"] = torch.tensor(1.5)
            torch.save(payload, self.path)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                load_training_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=self.cursor,
                    training_config=self.training,
                    corpus_fingerprint="a" * 64,
                )
            for key, expected in optimizer_before["state"].items():
                for name, tensor in expected.items():
                    self.assertTrue(
                        torch.equal(
                            self.optimizer.state_dict()["state"][key][name], tensor
                        )
                    )

    def test_malformed_live_adamw_state_cannot_replace_a_checkpoint(self) -> None:
        self.populate_optimizer()
        self.save()
        original = self.path.read_bytes()
        state = self.optimizer.state[next(self.model.parameters())]
        state["exp_avg"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            self.save(2)
        self.assertEqual(self.path.read_bytes(), original)

    def test_amsgrad_state_restores_and_performs_a_valid_update(self) -> None:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), amsgrad=True)
        self.populate_optimizer()
        self.save()
        load_training_checkpoint(
            self.path,
            model=self.model,
            optimizer=self.optimizer,
            cursor=self.cursor,
            training_config=self.training,
            corpus_fingerprint="a" * 64,
        )
        self.populate_optimizer()
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))

    def test_amsgrad_missing_maximum_is_rejected_before_resume(self) -> None:
        self.optimizer = torch.optim.AdamW(self.model.parameters(), amsgrad=True)
        self.populate_optimizer()
        self.save()
        payload = torch.load(self.path, weights_only=True)
        state = next(iter(payload["optimizer_state"]["state"].values()))
        del state["max_exp_avg_sq"]
        torch.save(payload, self.path)
        with self.assertRaises(ValueError):
            load_training_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                cursor=self.cursor,
                training_config=self.training,
                corpus_fingerprint="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
