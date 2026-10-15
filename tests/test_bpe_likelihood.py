import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_corpus import build_bpe_corpus
from ai_journey.bpe_likelihood import score_documents, summarize_likelihood
from ai_journey.transformer_lab import DecoderLanguageModel, TransformerConfig


class BPELikelihoodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.corpus = build_bpe_corpus(
            ("banana banana",), ("ab🙂", "c"), vocab_size=260
        )
        self.model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=self.corpus.vocab_size,
                block_size=3,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
                dropout=0.2,
            )
        )

    def test_uniform_oracle_counts_each_content_target_and_eos_once(self):
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
        docs = score_documents(
            self.model, self.corpus.codec, ("ab🙂", "c"), batch_size=2
        )
        summary = summarize_likelihood(docs)
        tokens = sum(len(self.corpus.codec.encode(text)) for text in ("ab🙂", "c"))
        self.assertEqual(summary["utf8_bytes"], 7)
        self.assertEqual(summary["content_tokens"], tokens)
        self.assertAlmostEqual(
            summary["content_nll"], tokens * math.log(self.corpus.vocab_size)
        )
        self.assertAlmostEqual(summary["eos_nll"], 2 * math.log(self.corpus.vocab_size))
        self.assertAlmostEqual(
            summary["content_bits_per_byte"],
            tokens * math.log2(self.corpus.vocab_size) / 7,
        )

    def test_matches_independent_one_target_oracle_across_context_rollover(self):
        self.model.eval()
        text = "ab🙂"
        eos = self.corpus.vocab_size - 1
        sequence = [eos, *self.corpus.codec.encode(text), eos]
        expected = []
        with torch.no_grad():
            for end in range(1, len(sequence)):
                context = torch.tensor([sequence[max(0, end - 3) : end]])
                logits, _ = self.model(context)
                expected.append(
                    float(
                        F.cross_entropy(
                            logits[:, -1].double(), torch.tensor([sequence[end]])
                        )
                    )
                )
        for batch in (1, 2, 64):
            score = score_documents(
                self.model, self.corpus.codec, (text,), batch_size=batch
            )[0]
            self.assertAlmostEqual(
                score.content_nll, math.fsum(expected[:-1]), places=6
            )
            self.assertAlmostEqual(score.eos_nll, expected[-1], places=6)

    def test_documents_are_isolated_and_modes_rng_weights_survive(self):
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        rng = torch.get_rng_state().clone()
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        pair = score_documents(self.model, self.corpus.codec, ("ab🙂", "c"))
        alone = score_documents(self.model, self.corpus.codec, ("c",))
        self.assertEqual(pair[1], alone[0])
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[key]))
        with patch.object(self.model, "forward", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):
                score_documents(self.model, self.corpus.codec, ("c",))
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_rejects_marker_injection_invalid_controls_and_nonfinite_logits(self):
        for batch in (True, 0, 129):
            with self.assertRaises(TokenizerError):
                score_documents(self.model, self.corpus.codec, ("c",), batch_size=batch)
        with self.assertRaises(TokenizerError):
            score_documents(self.model, self.corpus.codec, ("<|endoftext|>",))
        with torch.no_grad():
            self.model.lm_head.weight.fill_(float("nan"))
        with self.assertRaises(TokenizerError):
            score_documents(self.model, self.corpus.codec, ("c",))
