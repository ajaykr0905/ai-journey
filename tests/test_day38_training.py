from __future__ import annotations

import hashlib
import random
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.day38_training import (  # noqa: E402
    Day38DataConfig,
    Day38TrainingConfig,
    build_day38_report,
    prepare_bounded_corpus,
    run_day38_experiment,
    verify_day38_report,
    write_day38_report,
)
from ai_journey.gpt_bigram import CorpusSource, GPTBigramError  # noqa: E402
from ai_journey.transformer_lab import TransformerLabError  # noqa: E402


def source_for(text: str) -> CorpusSource:
    payload = text.encode("utf-8")
    return CorpusSource(
        name="test corpus",
        url="https://example.test/corpus.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        license="test-only",
    )


class Day38DataTests(unittest.TestCase):
    def test_bounded_partitions_are_disjoint_and_ordered(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz" * 8
        config = Day38DataConfig(
            block_size=4,
            train_token_limit=40,
            validation_token_limit=12,
            validation_fraction=0.25,
        )

        prepared = prepare_bounded_corpus(text, source_for(text), config)

        encoded = prepared.corpus.codec.encode(text)
        self.assertEqual(prepared.split_index, 156)
        self.assertEqual(prepared.full_train_token_count, 156)
        self.assertEqual(prepared.full_validation_token_count, 52)
        self.assertEqual(prepared.used_train_token_count, 40)
        self.assertEqual(prepared.used_validation_token_count, 12)
        self.assertEqual(prepared.corpus.train_tokens.tolist(), list(encoded[:40]))
        self.assertEqual(
            prepared.corpus.validation_tokens.tolist(), list(encoded[156:168])
        )

    def test_source_digest_is_verified_before_tokenization(self) -> None:
        text = "abcdefghij" * 8
        source = source_for(text)

        with self.assertRaises(GPTBigramError):
            prepare_bounded_corpus(
                text + "x",
                source,
                Day38DataConfig(
                    block_size=4,
                    train_token_limit=40,
                    validation_token_limit=12,
                ),
            )


class Day38ExperimentTests(unittest.TestCase):
    text = "First Citizen:\nWe know the model.\n" * 20

    def config(self) -> Day38TrainingConfig:
        return Day38TrainingConfig(
            data=Day38DataConfig(
                block_size=4,
                train_token_limit=320,
                validation_token_limit=80,
                validation_fraction=0.2,
            ),
            embedding_dim=8,
            head_count=2,
            layer_count=1,
            dropout=0.1,
            steps=8,
            batch_size=4,
            learning_rate=1e-2,
            seed=38,
            sample_prompt="\n",
            sample_tokens=12,
            sample_temperature=0.8,
            sample_top_k=5,
            sample_seed=3800,
        )

    def run_experiment(self, root: Path):
        return run_day38_experiment(
            self.text,
            source_for(self.text),
            self.config(),
            checkpoint_path=root / "day38.pt",
        )

    def test_full_transformer_training_and_checkpoint_sample_are_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.run_experiment(Path(first_dir))
            second = self.run_experiment(Path(second_dir))

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertLess(first.final_train_nll, first.initial_train_nll)
        self.assertEqual(first.completed_steps, 8)
        self.assertEqual(len(first.trace), 8)
        self.assertEqual(len(first.sample_token_ids), 13)
        self.assertTrue(first.sample.startswith("\n"))

    def test_completed_checkpoint_replays_without_additional_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "day38.pt"
            first = run_day38_experiment(
                self.text,
                source_for(self.text),
                self.config(),
                checkpoint_path=checkpoint,
            )
            resumed = run_day38_experiment(
                self.text,
                source_for(self.text),
                self.config(),
                checkpoint_path=checkpoint,
                resume=True,
            )

        self.assertEqual(resumed.trace, ())
        self.assertEqual(resumed.model_fingerprint, first.model_fingerprint)
        self.assertEqual(resumed.sample, first.sample)
        self.assertEqual(resumed.final_validation_nll, first.final_validation_nll)

    def test_experiment_preserves_caller_random_state(self) -> None:
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()

        with tempfile.TemporaryDirectory() as directory:
            self.run_experiment(Path(directory))

        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_report_is_self_verifying_and_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = self.run_experiment(root)
            report = build_day38_report(experiment, elapsed_seconds=1.25)
            output = root / "nested" / "day38.json"
            write_day38_report(output, report)
            written = json.loads(output.read_text(encoding="utf-8"))

        verify_day38_report(written)
        self.assertEqual(written["runtime"]["device"], "cpu")
        self.assertEqual(written["finding"], "training-and-validation-nll-decreased")
        self.assertIn("bounded CPU character-level experiment", written["limitations"])

    def test_report_rejects_tampered_or_nonfinite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = build_day38_report(
                self.run_experiment(Path(directory)), elapsed_seconds=1.0
            )
        report["experiment"]["final_validation_nll"] += 0.5
        with self.assertRaisesRegex(TransformerLabError, "fingerprint mismatch"):
            verify_day38_report(report)

        with tempfile.TemporaryDirectory() as directory:
            report = build_day38_report(
                self.run_experiment(Path(directory)), elapsed_seconds=1.0
            )
        report["runtime"]["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(TransformerLabError, "finite JSON"):
            verify_day38_report(report)

    def test_budgets_must_exceed_context_window(self) -> None:
        with self.assertRaises(TransformerLabError):
            Day38DataConfig(
                block_size=8,
                train_token_limit=8,
                validation_token_limit=9,
            )

    def test_small_source_split_is_rejected(self) -> None:
        text = "abcdefghij"
        with self.assertRaises(TransformerLabError):
            prepare_bounded_corpus(
                text,
                source_for(text),
                Day38DataConfig(
                    block_size=4,
                    train_token_limit=8,
                    validation_token_limit=8,
                    validation_fraction=0.1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
