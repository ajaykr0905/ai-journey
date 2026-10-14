import unittest
from unittest.mock import patch

from ai_journey.bpe import TokenizerError
from ai_journey.bpe_artifact import BPEArtifact
from ai_journey.bpe_corpus import build_bpe_corpus
from ai_journey.bpe_serving_benchmark import benchmark_bpe_serving
from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    TransformerConfig,
    model_fingerprint,
)


class PairedUtf8BenchmarkTests(unittest.TestCase):
    def setUp(self):
        corpus = build_bpe_corpus(("hello hello नमस्ते",), ("held out",), vocab_size=270)
        model = DecoderLanguageModel(
            TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
        ).double()
        self.artifact = BPEArtifact(
            model,
            corpus,
            {"experiment": {"model_fingerprint": model_fingerprint(model)}},
        )

    def test_paired_tokens_and_weight_identity_gate_every_timing(self):
        report = benchmark_bpe_serving(self.artifact, "hello", repeats=2)
        self.assertEqual(report["model_sha256"], model_fingerprint(self.artifact.model))
        self.assertEqual(len(report["elapsed_ns"]["kv-cache"]), 2)
        self.assertGreater(report["generated_tokens"], 0)
        self.assertTrue(self.artifact.model.training)

    def test_regressed_cache_output_prevents_a_benchmark_report(self):
        from ai_journey.bpe_serving import generate_utf8

        def corrupted(*args, **kwargs):
            sample = generate_utf8(*args, **kwargs)
            if kwargs.get("cached"):
                sample["text"] += "changed"
            return sample

        with patch(
            "ai_journey.bpe_serving_benchmark.generate_utf8", side_effect=corrupted
        ), self.assertRaisesRegex(TokenizerError, "parity"):
            benchmark_bpe_serving(self.artifact, "hello", repeats=1)
