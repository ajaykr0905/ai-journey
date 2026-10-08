from __future__ import annotations

import math
import json
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from ai_journey.gpt_bigram import (
    BigramExperiment,
    BigramLanguageModel,
    BigramTrainingConfig,
    CharacterVocabulary,
    CorpusSource,
    GPTBigramError,
    TINY_SHAKESPEARE,
    TrainingStep,
    WindowBatcher,
    build_checkpoint_payload,
    build_experiment_report,
    evaluate_partition,
    fetch_verified_corpus,
    generate_tokens,
    load_checkpoint,
    load_verified_corpus,
    model_fingerprint,
    next_token_loss,
    restore_checkpoint,
    run_bigram_experiment,
    save_checkpoint,
    tokenize_and_split,
    train_steps,
    validate_corpus_bytes,
    verify_experiment_report,
    write_experiment_report,
)


def source_for(payload: bytes) -> CorpusSource:
    return CorpusSource(
        "fixture",
        "https://example.test/fixture.txt",
        sha256(payload).hexdigest(),
        len(payload),
        "CC0-1.0",
    )


class CorpusSourceTests(unittest.TestCase):
    def test_tiny_shakespeare_source_is_content_addressed(self) -> None:
        self.assertEqual(TINY_SHAKESPEARE.byte_count, 1_115_394)
        self.assertEqual(len(TINY_SHAKESPEARE.sha256), 64)
        self.assertTrue(TINY_SHAKESPEARE.url.startswith("https://"))

    def test_source_rejects_unpinned_or_insecure_metadata(self) -> None:
        with self.assertRaisesRegex(GPTBigramError, "HTTPS"):
            CorpusSource("corpus", "http://example.test/data", "0" * 64, 1, "CC0")
        with self.assertRaisesRegex(GPTBigramError, "sha256"):
            CorpusSource("corpus", "https://example.test/data", "ABC", 1, "CC0")
        with self.assertRaisesRegex(GPTBigramError, "positive"):
            CorpusSource("corpus", "https://example.test/data", "0" * 64, 0, "CC0")


class CorpusBytesTests(unittest.TestCase):
    def test_validates_and_decodes_exact_public_bytes(self) -> None:
        payload = "First Citizen:\nSpeak.\n".encode()
        self.assertEqual(
            validate_corpus_bytes(payload, source_for(payload)), payload.decode()
        )

    def test_rejects_truncation_and_content_substitution(self) -> None:
        payload = b"abcd"
        source = source_for(payload)
        with self.assertRaisesRegex(GPTBigramError, "byte count mismatch"):
            validate_corpus_bytes(payload[:-1], source)
        with self.assertRaisesRegex(GPTBigramError, "sha256 mismatch"):
            validate_corpus_bytes(b"abce", source)

    def test_loads_only_a_matching_local_snapshot(self) -> None:
        payload = b"To be, or not to be.\n"
        with TemporaryDirectory() as directory:
            path = Path(directory, "input.txt")
            path.write_bytes(payload)
            self.assertEqual(
                load_verified_corpus(path, source_for(payload)), payload.decode()
            )
            path.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(GPTBigramError, "byte count mismatch"):
                load_verified_corpus(path, source_for(payload))

    def test_downloads_verifies_and_atomically_caches_the_snapshot(self) -> None:
        payload = b"First Citizen:\nSpeak.\n"
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "input.txt")
            with patch("ai_journey.gpt_bigram.urlopen", return_value=BytesIO(payload)):
                text = fetch_verified_corpus(
                    path, source_for(payload), timeout_seconds=1
                )
            self.assertEqual(text, payload.decode())
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_failed_download_does_not_replace_existing_snapshot(self) -> None:
        payload = b"correct text"
        with TemporaryDirectory() as directory:
            path = Path(directory, "input.txt")
            path.write_bytes(b"preserve me")
            with patch(
                "ai_journey.gpt_bigram.urlopen", return_value=BytesIO(b"wrong bytes!")
            ):
                with self.assertRaisesRegex(GPTBigramError, "sha256 mismatch"):
                    fetch_verified_corpus(path, source_for(payload))
            self.assertEqual(path.read_bytes(), b"preserve me")


class CharacterVocabularyTests(unittest.TestCase):
    def test_round_trips_text_with_stable_sorted_ids(self) -> None:
        vocabulary = CharacterVocabulary.from_text("cab\nca")
        self.assertEqual(vocabulary.tokens, ("\n", "a", "b", "c"))
        encoded = vocabulary.encode("cab\n")
        self.assertEqual(encoded, (3, 1, 2, 0))
        self.assertEqual(vocabulary.decode(encoded), "cab\n")

    def test_rejects_unknown_characters_and_invalid_ids(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        with self.assertRaisesRegex(GPTBigramError, "unknown character"):
            vocabulary.encode("abd")
        with self.assertRaisesRegex(GPTBigramError, "out of range"):
            vocabulary.decode([3])


class CorpusSplitTests(unittest.TestCase):
    def test_uses_a_contiguous_held_out_suffix(self) -> None:
        text = "abcdefghij"
        vocabulary = CharacterVocabulary.from_text(text)
        split = tokenize_and_split(text, vocabulary, validation_fraction=0.2)
        self.assertEqual(split.split_index, 8)
        self.assertEqual(vocabulary.decode(split.train.tolist()), "abcdefgh")
        self.assertEqual(vocabulary.decode(split.validation.tolist()), "ij")
        self.assertEqual(split.train.dtype, split.validation.dtype)

    def test_rejects_partitions_too_short_for_next_token_targets(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abcd")
        with self.assertRaisesRegex(GPTBigramError, "at least two"):
            tokenize_and_split("abcd", vocabulary, validation_fraction=0.25)


class BigramTrainingConfigTests(unittest.TestCase):
    def test_accepts_explicit_bounded_controls(self) -> None:
        config = BigramTrainingConfig(
            steps=25,
            batch_size=8,
            block_size=4,
            learning_rate=0.03,
            seed=9,
            eval_interval=5,
            eval_batch_size=16,
        )
        self.assertEqual(config.steps, 25)
        self.assertEqual(config.seed, 9)

    def test_rejects_non_executable_controls(self) -> None:
        with self.assertRaisesRegex(GPTBigramError, "steps must be positive"):
            BigramTrainingConfig(steps=0)
        with self.assertRaisesRegex(GPTBigramError, "positive and finite"):
            BigramTrainingConfig(learning_rate=float("nan"))
        with self.assertRaisesRegex(TypeError, "seed must be an integer"):
            BigramTrainingConfig(seed=True)

    def test_rejects_invalid_sampling_controls(self) -> None:
        with self.assertRaisesRegex(GPTBigramError, "sample_tokens must be positive"):
            BigramTrainingConfig(sample_tokens=0)
        with self.assertRaisesRegex(GPTBigramError, "sample_temperature"):
            BigramTrainingConfig(sample_temperature=float("inf"))


class WindowBatcherTests(unittest.TestCase):
    def test_samples_aligned_next_token_windows(self) -> None:
        tokens = torch.arange(12, dtype=torch.long)
        inputs, targets = WindowBatcher(tokens, block_size=4, seed=34).sample(3)
        self.assertEqual(tuple(inputs.shape), (3, 4))
        self.assertTrue(torch.equal(targets[:, :-1], inputs[:, 1:]))
        self.assertTrue(torch.equal(targets[:, 0], inputs[:, 0] + 1))

    def test_same_seed_replays_the_same_batches(self) -> None:
        tokens = torch.arange(20, dtype=torch.long)
        first = WindowBatcher(tokens, block_size=5, seed=7).sample(4)
        second = WindowBatcher(tokens, block_size=5, seed=7).sample(4)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_restores_the_next_batch_exactly(self) -> None:
        batcher = WindowBatcher(torch.arange(30), block_size=6, seed=11)
        batcher.sample(2)
        state = batcher.rng_state()
        expected = batcher.sample(5)
        batcher.restore_rng_state(state)
        replayed = batcher.sample(5)
        self.assertTrue(torch.equal(expected[0], replayed[0]))
        self.assertTrue(torch.equal(expected[1], replayed[1]))

    def test_rng_state_is_returned_by_value(self) -> None:
        batcher = WindowBatcher(torch.arange(20), block_size=4, seed=5)
        state = batcher.rng_state()
        state.zero_()
        self.assertFalse(torch.equal(state, batcher.rng_state()))


class BigramLanguageModelTests(unittest.TestCase):
    def test_maps_each_input_id_to_one_next_token_logit_row(self) -> None:
        model = BigramLanguageModel(5, seed=34)
        token_ids = torch.tensor([[0, 2, 4], [4, 2, 0]])
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        self.assertTrue(torch.equal(logits[0, 0], logits[1, 2]))
        self.assertTrue(torch.equal(logits[0, 1], logits[1, 1]))

    def test_seeded_initialization_does_not_mutate_global_rng(self) -> None:
        torch.manual_seed(99)
        expected = torch.rand(3)
        torch.manual_seed(99)
        BigramLanguageModel(4, seed=12)
        actual = torch.rand(3)
        self.assertTrue(torch.equal(expected, actual))

    def test_uniform_logits_have_log_vocabulary_loss(self) -> None:
        model = BigramLanguageModel(4, seed=1)
        with torch.no_grad():
            model.token_embedding_table.weight.zero_()
        inputs = torch.tensor([[0, 1], [2, 3]])
        targets = torch.tensor([[1, 2], [3, 0]])
        self.assertAlmostEqual(
            next_token_loss(model, inputs, targets).item(), math.log(4)
        )

    def test_loss_rejects_misaligned_targets(self) -> None:
        model = BigramLanguageModel(3, seed=1)
        with self.assertRaisesRegex(GPTBigramError, "same rank-two shape"):
            next_token_loss(model, torch.tensor([[0, 1]]), torch.tensor([[1]]))

    def test_evaluation_covers_every_pair_without_sampling(self) -> None:
        model = BigramLanguageModel(3, seed=1)
        with torch.no_grad():
            model.token_embedding_table.weight.zero_()
        tokens = torch.tensor([0, 1, 2, 0, 2, 1])
        model.train()
        self.assertAlmostEqual(
            evaluate_partition(model, tokens, chunk_size=2), math.log(3), places=6
        )
        self.assertTrue(model.training)

    def test_evaluation_rejects_empty_pair_sets(self) -> None:
        model = BigramLanguageModel(3, seed=1)
        with self.assertRaisesRegex(GPTBigramError, "length at least two"):
            evaluate_partition(model, torch.tensor([0]))

    def test_training_updates_reduce_repetitive_sequence_loss(self) -> None:
        tokens = torch.tensor(([0, 1] * 40) + [0])
        model = BigramLanguageModel(2, seed=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        batcher = WindowBatcher(tokens, block_size=4, seed=4)
        initial = evaluate_partition(model, tokens)
        trace = train_steps(
            model,
            optimizer,
            batcher,
            start_step=0,
            steps=30,
            batch_size=8,
        )
        self.assertEqual(trace[0].step, 1)
        self.assertEqual(trace[-1].step, 30)
        self.assertTrue(all(isinstance(point, TrainingStep) for point in trace))
        self.assertLess(evaluate_partition(model, tokens), initial)

    def test_generation_is_seeded_bounded_and_mode_preserving(self) -> None:
        model = BigramLanguageModel(4, seed=2)
        model.train()
        first = generate_tokens(model, start_token_id=0, max_new_tokens=12, seed=9)
        second = generate_tokens(model, start_token_id=0, max_new_tokens=12, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 13)
        self.assertTrue(all(0 <= token_id < 4 for token_id in first))
        self.assertTrue(model.training)

    def test_generation_rejects_unbounded_or_invalid_controls(self) -> None:
        model = BigramLanguageModel(3, seed=2)
        with self.assertRaisesRegex(GPTBigramError, "max_new_tokens"):
            generate_tokens(model, start_token_id=0, max_new_tokens=0, seed=1)
        with self.assertRaisesRegex(GPTBigramError, "temperature"):
            generate_tokens(
                model, start_token_id=0, max_new_tokens=1, seed=1, temperature=0
            )

    def test_model_fingerprint_binds_exact_parameters(self) -> None:
        first = BigramLanguageModel(3, seed=8)
        second = BigramLanguageModel(3, seed=8)
        self.assertEqual(model_fingerprint(first), model_fingerprint(second))
        with torch.no_grad():
            second.token_embedding_table.weight[0, 0] += 1
        self.assertNotEqual(model_fingerprint(first), model_fingerprint(second))

    def test_checkpoint_payload_captures_complete_state_by_value(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        model = BigramLanguageModel(3, seed=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        batcher = WindowBatcher(torch.tensor([0, 1, 2, 0, 1]), block_size=2, seed=5)
        payload = build_checkpoint_payload(
            model,
            optimizer,
            batcher,
            step=7,
            vocabulary=vocabulary,
            source=source_for(b"abc"),
            config=BigramTrainingConfig(block_size=2),
        )
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["vocabulary"], ("a", "b", "c"))
        captured = payload["model_state"]["token_embedding_table.weight"].clone()
        with torch.no_grad():
            model.token_embedding_table.weight.zero_()
        self.assertTrue(
            torch.equal(
                captured, payload["model_state"]["token_embedding_table.weight"]
            )
        )

    def test_checkpoint_publication_is_complete_and_leaves_no_temporary(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        model = BigramLanguageModel(3, seed=4)
        payload = build_checkpoint_payload(
            model,
            torch.optim.AdamW(model.parameters()),
            WindowBatcher(torch.tensor([0, 1, 2, 0]), block_size=2, seed=5),
            step=0,
            vocabulary=vocabulary,
            source=source_for(b"abc"),
            config=BigramTrainingConfig(block_size=2),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "checkpoint.pt")
            save_checkpoint(path, payload)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["model_fingerprint"], payload["model_fingerprint"])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_checkpoint_loader_rejects_unknown_or_missing_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "bad.pt")
            torch.save({"format_version": 1, "unknown": True}, path)
            with self.assertRaisesRegex(GPTBigramError, "fields do not match"):
                load_checkpoint(path)

    def test_checkpoint_loader_rejects_truncated_archives(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "truncated.pt")
            path.write_bytes(b"not a torch archive")
            with self.assertRaisesRegex(GPTBigramError, "unable to load"):
                load_checkpoint(path)

    def test_checkpoint_restore_enforces_source_and_recovers_exact_model(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        source = source_for(b"abc")
        config = BigramTrainingConfig(block_size=2)
        model = BigramLanguageModel(3, seed=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        batcher = WindowBatcher(torch.tensor([0, 1, 2, 0]), block_size=2, seed=5)
        payload = build_checkpoint_payload(
            model,
            optimizer,
            batcher,
            step=3,
            vocabulary=vocabulary,
            source=source,
            config=config,
        )
        expected_fingerprint = model_fingerprint(model)
        with torch.no_grad():
            model.token_embedding_table.weight.zero_()
        self.assertEqual(
            restore_checkpoint(
                payload,
                model,
                optimizer,
                batcher,
                vocabulary=vocabulary,
                source=source,
                config=config,
            ),
            3,
        )
        self.assertEqual(model_fingerprint(model), expected_fingerprint)
        wrong_source = source_for(b"abd")
        with self.assertRaisesRegex(GPTBigramError, "source digest"):
            restore_checkpoint(
                payload,
                model,
                optimizer,
                batcher,
                vocabulary=vocabulary,
                source=wrong_source,
                config=config,
            )

    def test_interrupted_training_replays_the_uninterrupted_control(self) -> None:
        vocabulary = CharacterVocabulary.from_text("abc")
        tokens = torch.tensor(([0, 1, 2] * 20) + [0])
        source = source_for(b"abc")
        config = BigramTrainingConfig(
            steps=10, batch_size=6, block_size=3, learning_rate=0.02, seed=17
        )

        control_model = BigramLanguageModel(3, seed=config.seed)
        control_optimizer = torch.optim.AdamW(
            control_model.parameters(), lr=config.learning_rate
        )
        control_batcher = WindowBatcher(
            tokens, block_size=config.block_size, seed=config.seed + 1
        )
        control_trace = train_steps(
            control_model,
            control_optimizer,
            control_batcher,
            start_step=0,
            steps=config.steps,
            batch_size=config.batch_size,
        )

        first_model = BigramLanguageModel(3, seed=config.seed)
        first_optimizer = torch.optim.AdamW(
            first_model.parameters(), lr=config.learning_rate
        )
        first_batcher = WindowBatcher(
            tokens, block_size=config.block_size, seed=config.seed + 1
        )
        first_trace = train_steps(
            first_model,
            first_optimizer,
            first_batcher,
            start_step=0,
            steps=4,
            batch_size=config.batch_size,
        )
        payload = build_checkpoint_payload(
            first_model,
            first_optimizer,
            first_batcher,
            step=4,
            vocabulary=vocabulary,
            source=source,
            config=config,
        )

        resumed_model = BigramLanguageModel(3, seed=999)
        resumed_optimizer = torch.optim.AdamW(
            resumed_model.parameters(), lr=config.learning_rate
        )
        resumed_batcher = WindowBatcher(tokens, block_size=config.block_size, seed=999)
        restored_step = restore_checkpoint(
            payload,
            resumed_model,
            resumed_optimizer,
            resumed_batcher,
            vocabulary=vocabulary,
            source=source,
            config=config,
        )
        resumed_trace = train_steps(
            resumed_model,
            resumed_optimizer,
            resumed_batcher,
            start_step=restored_step,
            steps=config.steps - restored_step,
            batch_size=config.batch_size,
        )
        self.assertEqual(first_trace, control_trace[:4])
        self.assertEqual(resumed_trace, control_trace[4:])
        self.assertEqual(
            model_fingerprint(resumed_model), model_fingerprint(control_model)
        )

    def test_complete_experiment_measures_training_and_held_out_loss(self) -> None:
        text = ("abcabc\n" * 80) + ("cab\n" * 20)
        source = source_for(text.encode())
        config = BigramTrainingConfig(
            steps=30,
            batch_size=8,
            block_size=4,
            learning_rate=0.05,
            seed=34,
            sample_tokens=20,
        )
        experiment, model, optimizer, batcher, vocabulary = run_bigram_experiment(
            text, source, config
        )
        self.assertIsInstance(experiment, BigramExperiment)
        self.assertLess(experiment.final_train_nll, experiment.initial_train_nll)
        self.assertEqual(experiment.completed_step, 30)
        self.assertEqual(len(experiment.trace), 30)
        self.assertEqual(len(experiment.sample), 21)
        self.assertEqual(experiment.model_fingerprint, model_fingerprint(model))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(batcher, WindowBatcher)
        self.assertEqual(experiment.vocabulary_size, len(vocabulary.tokens))

        report = build_experiment_report(experiment)
        verify_experiment_report(report)
        tampered = dict(report)
        tampered["sample"] = "changed"
        with self.assertRaisesRegex(GPTBigramError, "fingerprint"):
            verify_experiment_report(tampered)
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "report.json")
            write_experiment_report(path, report)
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
