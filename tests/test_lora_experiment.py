import copy
from dataclasses import replace
import math
import random
import unittest

import numpy as np
import torch

from ai_journey.lora_data import (
    Document,
    LoRAExperimentError,
    synthetic_fixture,
    training_batches,
)
from ai_journey.lora_experiment import (
    ComparisonConfig,
    evaluate_heldout,
    isolated_cpu_runtime,
    train_on_batches,
    run_dense_comparison,
)
from ai_journey.transformer_lab import DecoderLanguageModel, model_fingerprint


def tiny_config(**kwargs):
    return ComparisonConfig(
        **{
            "block_size": 3,
            "embedding_dim": 4,
            "head_count": 1,
            "batch_size": 2,
            "pretrain_steps": 1,
            "finetune_steps": 1,
            **kwargs,
        }
    )


class LoRAControlsTests(unittest.TestCase):
    def test_rejects_unpaired_invalid_and_excessive_work_controls(self):
        config = ComparisonConfig()
        for field, value in (
            ("seeds", (1,)),
            ("seeds", (1, 1)),
            ("seeds", [1, 2]),
            ("seeds", (True, 2)),
            ("seeds", (-1, 2)),
            ("embedding_dim", 7),
            ("block_size", 33),
            ("batch_size", True),
            ("finetune_steps", 0),
            ("pretrain_learning_rate", float("nan")),
            ("adapter_scale", float("inf")),
            ("finetune_learning_rate", True),
            ("evaluation_batch_size", 65),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(
                LoRAExperimentError
            ):
                replace(config, **{field: value})
        with self.assertRaisesRegex(LoRAExperimentError, "work budget"):
            ComparisonConfig(
                block_size=32,
                embedding_dim=32,
                batch_size=8,
                pretrain_steps=64,
                finetune_steps=64,
            )

    def test_runtime_restores_rng_dtype_threads_and_determinism_even_on_failure(self):
        for fail in (False, True):
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            controls = (
                torch.get_num_threads(),
                torch.get_default_dtype(),
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            try:
                with isolated_cpu_runtime(37):
                    random.random()
                    np.random.rand()
                    torch.rand(2)
                    self.assertEqual(torch.get_default_dtype(), torch.float64)
                    self.assertEqual(torch.get_num_threads(), 1)
                    if fail:
                        raise RuntimeError("injected failure")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(random.getstate(), python_state)
            current_np = np.random.get_state()
            self.assertEqual(current_np[0], numpy_state[0])
            np.testing.assert_array_equal(current_np[1], numpy_state[1])
            self.assertEqual(current_np[2:], numpy_state[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertEqual(
                controls,
                (
                    torch.get_num_threads(),
                    torch.get_default_dtype(),
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                ),
            )


class LoRADensePrimitivesTests(unittest.TestCase):
    def test_heldout_uniform_model_has_analytical_nll_and_preserves_modes(self):
        split = synthetic_fixture()
        with isolated_cpu_runtime(11):
            model = DecoderLanguageModel(
                tiny_config().model_config(len(split.codec().tokens))
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            model.train()
            model.blocks[0].attention.eval()
            modes = [module.training for module in model.modules()]
            before = model_fingerprint(model)
            documents = (Document("a", "calm"), Document("b", "cedar.\n"))
            score = evaluate_heldout(model, documents, split.codec(), batch_size=2)
            self.assertEqual(score["target_tokens"], 9)
            self.assertAlmostEqual(
                score["mean_nll"], math.log(len(split.codec().tokens)), places=12
            )
            self.assertAlmostEqual(
                score["perplexity"], len(split.codec().tokens), places=12
            )
            self.assertEqual(before, model_fingerprint(model))
            self.assertEqual(modes, [module.training for module in model.modules()])

    def test_batched_holdout_matches_individual_document_local_predictions(self):
        split = synthetic_fixture()
        with isolated_cpu_runtime(11):
            model = DecoderLanguageModel(
                tiny_config().model_config(len(split.codec().tokens))
            )
            first = evaluate_heldout(
                model, split.validation, split.codec(), batch_size=1
            )
            batched = evaluate_heldout(
                model, split.validation, split.codec(), batch_size=32
            )
            self.assertEqual(first["target_tokens"], 376)
            self.assertAlmostEqual(first["nll_sum"], batched["nll_sum"], places=11)

    def test_dense_updates_and_frozen_control_consume_the_identical_schedule(self):
        split = synthetic_fixture()
        controls = tiny_config()
        with isolated_cpu_runtime(11):
            model = DecoderLanguageModel(
                controls.model_config(len(split.codec().tokens))
            )
            frozen = copy.deepcopy(model).requires_grad_(False)
            before = model_fingerprint(model)
            batches = training_batches(
                split.train, split.codec(), block_size=3, batch_size=2, steps=2, seed=1
            )
            dense_trace = train_on_batches(model, batches, learning_rate=0.01)
            frozen_trace = train_on_batches(frozen, batches, learning_rate=0.01)
            self.assertNotEqual(model_fingerprint(model), before)
            self.assertEqual(model_fingerprint(frozen), before)
            self.assertEqual(
                dense_trace["consumed_target_tokens"],
                frozen_trace["consumed_target_tokens"],
            )
            self.assertEqual(dense_trace["optimizer_steps"], 2)
            self.assertEqual(frozen_trace["optimizer_steps"], 0)
            self.assertTrue(all(math.isfinite(loss) for loss in dense_trace["losses"]))
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonfinite_parameters_fail_before_optimization_or_scoring(self):
        split = synthetic_fixture()
        with isolated_cpu_runtime(11):
            model = DecoderLanguageModel(
                tiny_config().model_config(len(split.codec().tokens))
            )
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] = float("nan")
            with self.assertRaisesRegex(LoRAExperimentError, "finite float64"):
                evaluate_heldout(model, split.validation, split.codec())
            with self.assertRaisesRegex(LoRAExperimentError, "finite float64"):
                train_on_batches(model, (), learning_rate=0.01)


class PairedDenseComparisonTests(unittest.TestCase):
    def test_each_seed_pretrains_once_and_forks_identical_dense_weights(self):
        result = run_dense_comparison(tiny_config())
        self.assertEqual([row["seed"] for row in result["seeds"]], [11, 29])
        self.assertNotEqual(
            result["seeds"][0]["pretrained_sha256"],
            result["seeds"][1]["pretrained_sha256"],
        )
        for row in result["seeds"]:
            self.assertNotEqual(row["initial_sha256"], row["pretrained_sha256"])
            self.assertEqual(row["pretrained_sha256"], row["source_after_sha256"])
            frozen, dense = row["trials"]
            self.assertEqual(frozen["initial_sha256"], row["pretrained_sha256"])
            self.assertEqual(dense["initial_sha256"], row["pretrained_sha256"])
            self.assertEqual(frozen["initial_sha256"], frozen["final_sha256"])
            self.assertNotEqual(dense["initial_sha256"], dense["final_sha256"])
            self.assertEqual(frozen["parameters"]["trainable"], 0)
            self.assertEqual(
                dense["parameters"]["total"], dense["parameters"]["trainable"]
            )
            for trial in row["trials"]:
                self.assertEqual(
                    trial["training"]["consumed_target_tokens"],
                    row["adaptation_schedule"]["target_tokens"],
                )
                self.assertEqual(
                    trial["heldout"]["target_tokens"],
                    result["heldout"]["target_tokens"],
                )
                self.assertEqual(
                    trial["frozen_before_sha256"], trial["frozen_after_sha256"]
                )

    def test_dense_comparison_is_exactly_reproducible(self):
        self.assertEqual(
            run_dense_comparison(tiny_config()), run_dense_comparison(tiny_config())
        )
