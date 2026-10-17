import copy
import math
import unittest

import torch

from ai_journey.lora_experiment import isolated_cpu_runtime
from ai_journey.sft_data import (
    SFTDataError,
    authored_chat_fixture,
    sft_training_batches,
)
from ai_journey.sft_experiment import train_sft_batches
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
)


def tiny_model(split):
    return DecoderLanguageModel(
        TransformerConfig(
            vocab_size=len(split.codec().tokens),
            block_size=16,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            dropout=0.0,
        )
    )


class SFTTrainingTests(unittest.TestCase):
    def test_dense_updates_and_frozen_control_use_exact_masked_exposure(self):
        split = authored_chat_fixture()
        batches = sft_training_batches(
            split.train,
            split.codec(),
            block_size=16,
            batch_size=2,
            steps=3,
            seed=43,
        )
        with isolated_cpu_runtime(43):
            dense = tiny_model(split)
            frozen = copy.deepcopy(dense).requires_grad_(False)
            dense_before = model_fingerprint(dense)
            frozen_before = model_fingerprint(frozen)
            dense.train()
            dense.blocks[0].attention.eval()
            modes = [module.training for module in dense.modules()]
            dense_trace = train_sft_batches(dense, batches, learning_rate=0.01)
            frozen_trace = train_sft_batches(frozen, batches, learning_rate=0.01)
            self.assertNotEqual(model_fingerprint(dense), dense_before)
            self.assertEqual(model_fingerprint(frozen), frozen_before)
            self.assertEqual(modes, [module.training for module in dense.modules()])
            self.assertEqual(dense_trace["input_tokens"], 96)
            self.assertEqual(
                dense_trace["supervised_tokens"],
                sum(row.supervised_tokens for batch in batches for row in batch),
            )
            self.assertEqual(
                dense_trace["supervised_tokens"], frozen_trace["supervised_tokens"]
            )
            self.assertEqual(dense_trace["optimizer_steps"], 3)
            self.assertEqual(frozen_trace["optimizer_steps"], 0)
            self.assertTrue(all(math.isfinite(loss) for loss in dense_trace["losses"]))
            self.assertTrue(
                all(parameter.grad is None for parameter in dense.parameters())
            )

    def test_rejects_invalid_controls_batches_and_nonfinite_models(self):
        split = authored_chat_fixture()
        batches = sft_training_batches(
            split.train,
            split.codec(),
            block_size=16,
            batch_size=2,
            steps=1,
            seed=43,
        )
        with isolated_cpu_runtime(43):
            for value in (0, True, float("nan"), float("inf"), 0.2):
                with self.subTest(value=value), self.assertRaises(SFTDataError):
                    train_sft_batches(tiny_model(split), batches, learning_rate=value)
            for invalid in ((), list(batches), (batches[0], (batches[0][0],))):
                with self.subTest(invalid=invalid), self.assertRaises(SFTDataError):
                    train_sft_batches(tiny_model(split), invalid, learning_rate=0.01)
            model = tiny_model(split)
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float("nan")
            with self.assertRaisesRegex(SFTDataError, "finite floating-point"):
                train_sft_batches(model, batches, learning_rate=0.01)


if __name__ == "__main__":
    unittest.main()
