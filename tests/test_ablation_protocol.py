from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ai_journey.ablation_protocol import (
    AblationArm,
    ControlledAblation,
    build_ablation_report,
    evaluate_ablation,
    load_ablation_protocol,
    run_ablation,
    verify_ablation_report,
    write_ablation_report,
)
from ai_journey.transformer_lab import (
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
)


class ControlledAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TransformerConfig(
            vocab_size=12,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
        )
        self.training = TrainingConfig(
            steps=3,
            batch_size=4,
            learning_rate=3e-3,
            seed=31,
        )

    def protocol(self, **overrides: object) -> ControlledAblation:
        arms = (
            AblationArm("baseline", self.model, self.training),
            AblationArm(
                "low-rate",
                self.model,
                replace(self.training, learning_rate=1e-3),
            ),
            AblationArm(
                "high-rate",
                self.model,
                replace(self.training, learning_rate=6e-3),
            ),
        )
        values: dict[str, object] = {
            "name": "learning-rate-sweep",
            "hypothesis": "The baseline rate minimizes validation NLL.",
            "independent_variable": "training.learning_rate",
            "primary_metric": "validation_nll",
            "expected_direction": "lower",
            "minimum_effect": 0.01,
            "baseline_label": "baseline",
            "trial_seeds": (31, 32),
            "arms": arms,
        }
        values.update(overrides)
        return ControlledAblation(**values)  # type: ignore[arg-type]

    def test_protocol_records_hypothesis_arms_and_fixed_controls(self) -> None:
        protocol = self.protocol()
        payload = protocol.to_dict()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["baseline_label"], "baseline")
        self.assertEqual(len(payload["arms"]), 3)
        self.assertNotIn("training.learning_rate", payload["fixed_controls"])
        self.assertEqual(payload["fixed_controls"]["training.batch_size"], 4)
        self.assertEqual(payload["fixed_controls"]["paired_trial_seeds"], [31, 32])
        self.assertEqual(len(protocol.fingerprint()), 64)

    def test_fingerprint_is_stable_and_sensitive_to_the_hypothesis(self) -> None:
        protocol = self.protocol()
        self.assertEqual(protocol.fingerprint(), self.protocol().fingerprint())
        self.assertNotEqual(
            protocol.fingerprint(),
            self.protocol(hypothesis="A different prediction.").fingerprint(),
        )

    def test_rejects_an_undeclared_second_variable(self) -> None:
        arms = list(self.protocol().arms)
        arms[1] = replace(
            arms[1],
            training_config=replace(
                arms[1].training_config,
                batch_size=8,
            ),
        )
        with self.assertRaisesRegex(TransformerLabError, "training.batch_size"):
            self.protocol(arms=tuple(arms))

    def test_rejects_a_declared_variable_that_does_not_change(self) -> None:
        with self.assertRaisesRegex(
            TransformerLabError, "changed training.learning_rate"
        ):
            self.protocol(independent_variable="training.batch_size")

    def test_rejects_duplicate_independent_values(self) -> None:
        arms = self.protocol().arms
        duplicate = replace(arms[2], training_config=arms[1].training_config)
        with self.assertRaisesRegex(TransformerLabError, "values must be unique"):
            self.protocol(arms=(arms[0], arms[1], duplicate))

    def test_rejects_non_experimental_identity_and_seed_controls(self) -> None:
        with self.assertRaisesRegex(TransformerLabError, "paired controls"):
            self.protocol(independent_variable="model.vocab_size")
        with self.assertRaisesRegex(TransformerLabError, "paired controls"):
            self.protocol(independent_variable="training.seed")

    def test_rejects_duplicate_labels_and_trial_seeds(self) -> None:
        arms = self.protocol().arms
        with self.assertRaisesRegex(TransformerLabError, "labels must be unique"):
            self.protocol(arms=(arms[0], replace(arms[1], label="baseline")))
        with self.assertRaisesRegex(TransformerLabError, "trial_seeds must be unique"):
            self.protocol(trial_seeds=(31, 31))

    def test_rejects_trial_seeds_that_numpy_cannot_execute(self) -> None:
        # A predeclared protocol must be usable before any arm begins training.
        for seed in (-1, 2**32):
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(TransformerLabError, "trial_seeds.*range"),
            ):
                self.protocol(trial_seeds=(seed,))

    def test_accepts_both_executable_seed_boundaries(self) -> None:
        protocol = self.protocol(trial_seeds=(0, 2**32 - 1))
        self.assertEqual(protocol.trial_seeds, (0, 2**32 - 1))

    def test_loader_requires_an_integer_schema_version_not_boolean_or_float(
        self,
    ) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "day-31-learning-rate-ablation.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for version in (True, 1.0):
                with (
                    self.subTest(version=version),
                    self.assertRaisesRegex(TransformerLabError, "schema_version"),
                ):
                    path.write_text(
                        json.dumps({**payload, "schema_version": version}),
                        encoding="utf-8",
                    )
                    load_ablation_protocol(path, vocab_size=12)

    def test_rejects_invalid_text_thresholds_and_categories(self) -> None:
        cases = (
            ({"name": "Bad Name"}, "name must use"),
            ({"hypothesis": "  "}, "must not be blank"),
            ({"primary_metric": "accuracy"}, "primary_metric"),
            ({"expected_direction": "sideways"}, "expected_direction"),
            ({"minimum_effect": float("nan")}, "finite and non-negative"),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(TransformerLabError, message),
            ):
                self.protocol(**overrides)


class AblationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TransformerConfig(
            vocab_size=12,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
        )
        self.training = TrainingConfig(
            steps=1,
            batch_size=2,
            learning_rate=3e-3,
            seed=31,
        )

    def corpus(self, directory: str) -> TokenCorpus:
        path = Path(directory) / "corpus.txt"
        path.write_text("alpha\nbeta\ngamma\ndelta\n" * 8, encoding="utf-8")
        return TokenCorpus.from_path(path, validation_fraction=0.2, block_size=4)

    def compact_protocol(self, vocab_size: int) -> ControlledAblation:
        model = replace(self.model, vocab_size=vocab_size)
        training = self.training
        return ControlledAblation(
            name="learning-rate-sweep",
            hypothesis="The baseline rate minimizes validation NLL.",
            independent_variable="training.learning_rate",
            primary_metric="validation_nll",
            expected_direction="lower",
            minimum_effect=0.01,
            baseline_label="baseline",
            arms=(
                AblationArm("baseline", model, training),
                AblationArm(
                    "low-rate",
                    model,
                    replace(training, learning_rate=1e-3),
                ),
            ),
            trial_seeds=(31, 32),
        )

    def test_runner_pairs_seeds_and_produces_deterministic_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = self.corpus(directory)
            protocol = self.compact_protocol(corpus.vocab_size)
            first = run_ablation(corpus, protocol)
            second = run_ablation(corpus, protocol)

        self.assertEqual(first.evidence_fingerprint(), second.evidence_fingerprint())
        self.assertEqual(len(first.trials), 4)
        self.assertEqual(len(first.summaries), 2)
        self.assertEqual(len(first.contrasts), 1)
        for seed in protocol.trial_seeds:
            paired = [trial for trial in first.trials if trial.seed == seed]
            self.assertEqual(
                len({trial.initial_model_fingerprint for trial in paired}), 1
            )
            self.assertEqual(
                len({trial.first_batch_fingerprint for trial in paired}), 1
            )
        contrast = first.contrasts[0]
        baseline = {
            trial.seed: trial.validation_nll
            for trial in first.trials
            if trial.arm_label == "baseline"
        }
        variant = {
            trial.seed: trial.validation_nll
            for trial in first.trials
            if trial.arm_label == "low-rate"
        }
        self.assertEqual(
            contrast.paired_deltas,
            tuple(variant[seed] - baseline[seed] for seed in protocol.trial_seeds),
        )
        self.assertEqual(
            first.to_dict()["evidence_fingerprint"], first.evidence_fingerprint()
        )

    def test_runner_restores_caller_random_state(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = self.corpus(directory)
            protocol = self.compact_protocol(corpus.vocab_size)
            torch.manual_seed(901)
            np.random.seed(902)
            torch_state = torch.get_rng_state().clone()
            numpy_state = np.random.get_state()
            deterministic = torch.are_deterministic_algorithms_enabled()

            run_ablation(corpus, protocol)

        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_state[1], after_numpy[1]))
        self.assertEqual(numpy_state[2:], after_numpy[2:])
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_runner_rejects_a_corpus_vocabulary_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = self.corpus(directory)
            protocol = self.compact_protocol(corpus.vocab_size + 1)
            with self.assertRaisesRegex(TransformerLabError, "vocabulary"):
                run_ablation(corpus, protocol)

    def test_report_preserves_controls_outcome_and_negative_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = self.corpus(directory)
            protocol = replace(
                self.compact_protocol(corpus.vocab_size), trial_seeds=(31,)
            )
            result = run_ablation(corpus, protocol)
            report = build_ablation_report(result)
            output = Path(directory) / "ablation.json"
            output.write_text("old report", encoding="utf-8")
            write_ablation_report(output, result)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotEqual(output.read_text(encoding="utf-8"), "old report")

        verify_ablation_report(report)
        verify_ablation_report(saved)
        evaluation = evaluate_ablation(result)
        self.assertTrue(evaluation.complete_pairs)
        self.assertTrue(evaluation.matched_initial_models)
        self.assertTrue(evaluation.matched_first_batches)
        self.assertIn(evaluation.outcome, {"supports", "contradicts", "inconclusive"})
        self.assertEqual(saved["evaluation"]["outcome"], evaluation.outcome)
        self.assertEqual(saved["report_fingerprint"], report["report_fingerprint"])

    def test_report_verification_rejects_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = self.corpus(directory)
            protocol = replace(
                self.compact_protocol(corpus.vocab_size), trial_seeds=(31,)
            )
            report = build_ablation_report(run_ablation(corpus, protocol))
        report["evaluation"]["outcome"] = "supports"
        with self.assertRaisesRegex(TransformerLabError, "does not match"):
            verify_ablation_report(report)


if __name__ == "__main__":
    unittest.main()
