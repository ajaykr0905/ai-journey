import copy
import math
import unittest
from dataclasses import asdict, replace

import torch

from ai_journey.lora_experiment import isolated_cpu_runtime
from ai_journey.sft_data import (
    SFTDataError,
    authored_chat_fixture,
    sft_training_batches,
)
from ai_journey.sft_experiment import (
    SFTComparisonConfig,
    evaluate_sft,
    run_sft_comparison,
    run_sft_dense_comparison,
    sft_config_dict,
    sft_config_from_dict,
    train_sft_batches,
)
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


class SFTComparisonControlTests(unittest.TestCase):
    def test_controls_round_trip_complete_protocol(self):
        config = SFTComparisonConfig()
        self.assertEqual(sft_config_from_dict(sft_config_dict(config)), config)
        self.assertEqual(config.model_config(47).vocab_size, 47)
        self.assertGreater(config.work_units, 0)

    def test_rejects_unpaired_invalid_unknown_and_excessive_controls(self):
        config = SFTComparisonConfig()
        for field, value in (
            ("seeds", (1,)),
            ("seeds", (1, 1)),
            ("seeds", [1, 2]),
            ("seeds", (True, 2)),
            ("block_size", 7),
            ("embedding_dim", 15),
            ("head_count", 0),
            ("batch_size", 17),
            ("base_steps", 0),
            ("sft_steps", 129),
            ("base_learning_rate", float("nan")),
            ("sft_learning_rate", True),
            ("adapter_scale", float("inf")),
            ("evaluation_batch_size", 129),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(SFTDataError),
            ):
                replace(config, **{field: value})
        with self.assertRaisesRegex(SFTDataError, "work budget"):
            SFTComparisonConfig(
                block_size=256,
                embedding_dim=64,
                batch_size=16,
                base_steps=128,
                sft_steps=128,
            )
        payload = asdict(config)
        payload["unexpected"] = 1
        with self.assertRaisesRegex(SFTDataError, "exactly all"):
            sft_config_from_dict(payload)


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


class SFTEvaluationTests(unittest.TestCase):
    def test_uniform_model_has_analytical_assistant_nll_and_restores_modes(self):
        split = authored_chat_fixture()
        with isolated_cpu_runtime(43):
            model = tiny_model(split)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            model.train()
            model.blocks[0].attention.eval()
            modes = [module.training for module in model.modules()]
            score = evaluate_sft(model, split.validation, split.codec(), batch_size=7)
            self.assertAlmostEqual(
                score["mean_nll"], math.log(len(split.codec().tokens)), places=12
            )
            self.assertAlmostEqual(
                score["perplexity"], len(split.codec().tokens), places=11
            )
            self.assertEqual(modes, [module.training for module in model.modules()])

    def test_batched_evaluation_matches_one_target_at_a_time(self):
        split = authored_chat_fixture()
        with isolated_cpu_runtime(43):
            model = tiny_model(split)
            individual = evaluate_sft(
                model, split.validation, split.codec(), batch_size=1
            )
            batched = evaluate_sft(
                model, split.validation, split.codec(), batch_size=64
            )
            self.assertEqual(individual["target_tokens"], batched["target_tokens"])
            self.assertAlmostEqual(individual["nll_sum"], batched["nll_sum"], places=11)

    def test_evaluation_rejects_bad_batch_size_and_nonfinite_parameters(self):
        split = authored_chat_fixture()
        with isolated_cpu_runtime(43):
            model = tiny_model(split)
            for value in (0, True, 129):
                with self.subTest(value=value), self.assertRaises(SFTDataError):
                    evaluate_sft(
                        model, split.validation, split.codec(), batch_size=value
                    )
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float("inf")
            with self.assertRaisesRegex(SFTDataError, "finite floating-point"):
                evaluate_sft(model, split.validation, split.codec())


class SFTDenseComparisonTests(unittest.TestCase):
    def tiny_config(self):
        return SFTComparisonConfig(
            block_size=16,
            embedding_dim=8,
            head_count=2,
            batch_size=2,
            base_steps=1,
            sft_steps=1,
            evaluation_batch_size=64,
        )

    def test_each_seed_warms_once_and_forks_identical_dense_models(self):
        result = run_sft_dense_comparison(self.tiny_config())
        self.assertEqual([row["seed"] for row in result["seeds"]], [43, 44])
        self.assertEqual(result["fixture"]["id"], "authored-weather-tools-v1")
        self.assertIn("same training conversations", result["limitations"])
        for seed in result["seeds"]:
            self.assertNotEqual(seed["initial_sha256"], seed["warmed_sha256"])
            self.assertEqual(seed["warmed_sha256"], seed["source_after_sha256"])
            frozen, full = seed["trials"]
            self.assertEqual([frozen["name"], full["name"]], ["frozen", "full"])
            self.assertEqual(frozen["initial_sha256"], seed["warmed_sha256"])
            self.assertEqual(full["initial_sha256"], seed["warmed_sha256"])
            self.assertEqual(frozen["initial_sha256"], frozen["final_sha256"])
            self.assertNotEqual(full["initial_sha256"], full["final_sha256"])
            self.assertEqual(frozen["parameters"]["trainable"], 0)
            self.assertEqual(
                full["parameters"]["total"], full["parameters"]["trainable"]
            )
            for trial in seed["trials"]:
                self.assertEqual(
                    trial["training"]["supervised_tokens"],
                    seed["sft_schedule"]["supervised_tokens"],
                )
                self.assertEqual(
                    trial["heldout"]["target_tokens"],
                    result["heldout"]["target_tokens"],
                )

    def test_dense_comparison_is_exactly_repeatable_and_rejects_wrong_config(self):
        config = self.tiny_config()
        self.assertEqual(
            run_sft_dense_comparison(config), run_sft_dense_comparison(config)
        )
        with self.assertRaisesRegex(SFTDataError, "SFTComparisonConfig"):
            run_sft_dense_comparison("bad")


class SFTLoRAComparisonTests(unittest.TestCase):
    def tiny_config(self):
        return SFTComparisonConfig(
            block_size=16,
            embedding_dim=8,
            head_count=2,
            batch_size=2,
            base_steps=1,
            sft_steps=1,
            evaluation_batch_size=64,
        )

    def test_lora_forks_preserve_bases_merge_exactly_and_reduce_trainable_capacity(
        self,
    ):
        result = run_sft_comparison(self.tiny_config())
        for seed in result["seeds"]:
            self.assertEqual(
                [trial["name"] for trial in seed["trials"]],
                ["frozen", "full", "lora-rank-1", "lora-rank-2"],
            )
            full = seed["trials"][1]
            for rank, trial in enumerate(seed["trials"][2:], start=1):
                self.assertEqual(trial["adapter"]["rank"], rank)
                self.assertEqual(trial["adapter"]["alpha"], rank * 2.0)
                self.assertEqual(trial["initial_function_max_abs_error"], 0)
                self.assertEqual(
                    trial["frozen_before_sha256"],
                    trial["frozen_after_sha256"],
                )
                self.assertLess(
                    trial["parameters"]["trainable"],
                    full["parameters"]["trainable"],
                )
                self.assertLessEqual(trial["merge"]["probe_max_abs_error"], 1e-10)
                self.assertEqual(
                    trial["merge"]["parameters"]["total"],
                    full["parameters"]["total"],
                )

    def test_complete_lora_comparison_is_exactly_repeatable(self):
        config = self.tiny_config()
        self.assertEqual(run_sft_comparison(config), run_sft_comparison(config))


if __name__ == "__main__":
    unittest.main()
