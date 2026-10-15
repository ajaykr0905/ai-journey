from dataclasses import replace
import random
import unittest

import numpy as np
import torch

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_model_ablation import VocabularyProtocol, run_vocabulary_ablation


class VocabularyAblationTests(unittest.TestCase):
    train = ("banana banana banana", "abracadabra abracadabra")
    heldout = ("banana unseen", "नमस्ते")
    protocol = VocabularyProtocol(
        vocab_sizes=(256, 258, 260), seeds=(1, 2), steps=2, block_size=3, batch_size=2
    )

    def test_paired_runs_match_initial_trunk_exposure_and_replay(self):
        first = run_vocabulary_ablation(self.train, self.heldout, self.protocol)
        second = run_vocabulary_ablation(self.train, self.heldout, self.protocol)
        self.assertEqual(len(first["trials"]), 6)
        for index, (left, right) in enumerate(zip(first["trials"], second["trials"])):
            self.assertEqual(
                left["final_model_fingerprint"], right["final_model_fingerprint"]
            )
            self.assertEqual(left["final"], right["final"])
            self.assertEqual(left["training_target_tokens"], 12)
            self.assertEqual(left["final"]["utf8_bytes"], 31)
            self.assertEqual(len(left["trace"]), 2)
            self.assertNotEqual(
                left["initial_model_fingerprint"], left["final_model_fingerprint"]
            )
            peers = first["trials"][(index // 3) * 3 : (index // 3 + 1) * 3]
            self.assertEqual(
                len({item["initial_trunk_fingerprint"] for item in peers}), 1
            )
        self.assertEqual(len({item["parameter_count"] for item in first["trials"]}), 3)

    def test_preserves_rng_and_deterministic_flags(self):
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        deterministic = torch.are_deterministic_algorithms_enabled()
        run_vocabulary_ablation(self.train, self.heldout, self.protocol)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_invalid_protocol_leakage_and_unrealized_arms_fail_before_training(self):
        for change in (
            {"seeds": (1, 1)},
            {"steps": True},
            {"vocab_sizes": (257, 258, 260)},
        ):
            with self.assertRaises(TokenizerError):
                replace(self.protocol, **change)
        with self.assertRaises(TokenizerError):
            run_vocabulary_ablation(self.train, self.train, self.protocol)
        with self.assertRaises(TokenizerError):
            run_vocabulary_ablation(("abcdefghi",), ("unseen",), self.protocol)
