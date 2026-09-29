from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_journey.activation_experiment import (
    default_activation_modules,
    render_activation_histograms,
    render_gradient_histograms,
    run_initialization_comparison,
    write_comparison_report,
)
from ai_journey.transformer_lab import (
    TokenCorpus,
    TrainingConfig,
    TransformerConfig,
    TransformerLabError,
)


class InitializationComparisonTests(unittest.TestCase):
    def _corpus(self, directory: str) -> TokenCorpus:
        path = Path(directory) / "corpus.txt"
        path.write_text("anna\naria\namara\n" * 8, encoding="utf-8")
        return TokenCorpus.from_path(path, block_size=4)

    def test_comparison_is_deterministic_and_changes_only_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            model_config = TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
                layer_count=1,
            )
            training_config = TrainingConfig(
                steps=2, batch_size=4, learning_rate=0.01, seed=25
            )
            first = run_initialization_comparison(
                corpus,
                model_config=model_config,
                training_config=training_config,
                stressed_initialization_std=0.8,
            )
            second = run_initialization_comparison(
                corpus,
                model_config=model_config,
                training_config=training_config,
                stressed_initialization_std=0.8,
            )
        self.assertEqual(first.to_dict(), second.to_dict())
        baseline, stressed = first.variants
        self.assertEqual(baseline.name, "baseline")
        self.assertEqual(stressed.name, "stressed")
        self.assertEqual(len(baseline.trace), 2)
        self.assertNotEqual(baseline.model_fingerprint, stressed.model_fingerprint)
        base_activation = baseline.initial_snapshot.activations[0].distribution
        stressed_activation = stressed.initial_snapshot.activations[0].distribution
        self.assertGreater(stressed_activation.rms, base_activation.rms)

    def test_default_modules_cover_each_transformer_block(self) -> None:
        config = TransformerConfig(vocab_size=5, layer_count=3)
        self.assertEqual(
            default_activation_modules(config),
            (
                "blocks.0.feed_forward.network.1",
                "blocks.1.feed_forward.network.1",
                "blocks.2.feed_forward.network.1",
            ),
        )

    def test_comparison_rejects_identical_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            config = TransformerConfig(
                vocab_size=corpus.vocab_size,
                block_size=4,
                embedding_dim=8,
                head_count=2,
            )
            with self.assertRaisesRegex(TransformerLabError, "differ"):
                run_initialization_comparison(
                    corpus,
                    model_config=config,
                    training_config=TrainingConfig(steps=1),
                    stressed_initialization_std=config.initialization_std,
                )

    def test_report_writer_is_stable_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_initialization_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
                stressed_initialization_std=0.8,
            )
            output = Path(directory) / "nested" / "report.json"
            write_comparison_report(output, result)
            first = output.read_bytes()
            write_comparison_report(output, result)
            second = output.read_bytes()
            payload = json.loads(second)
        self.assertEqual(first, second)
        self.assertEqual(
            [item["name"] for item in payload["variants"]], ["baseline", "stressed"]
        )
        self.assertFalse(output.with_name(".report.json.tmp").exists())

    def test_histogram_renderer_writes_stable_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_initialization_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
                stressed_initialization_std=0.8,
            )
            output = Path(directory) / "plots" / "activations.svg"
            render_activation_histograms(output, result)
            first = output.read_bytes()
            render_activation_histograms(output, result)
            second = output.read_bytes()
        self.assertEqual(first, second)
        self.assertIn(b"baseline initial", first)
        self.assertIn(b"stressed final", first)

    def test_histogram_renderer_rejects_uncaptured_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_initialization_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
            )
            with self.assertRaisesRegex(TransformerLabError, "not captured"):
                render_activation_histograms(
                    Path(directory) / "plot.svg", result, module_name="missing"
                )

    def test_gradient_renderer_writes_stable_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_initialization_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
                stressed_initialization_std=0.8,
            )
            output = Path(directory) / "plots" / "gradients.svg"
            parameter = "blocks.0.feed_forward.network.0.weight"
            render_gradient_histograms(output, result, parameter_name=parameter)
            first = output.read_bytes()
            render_gradient_histograms(output, result, parameter_name=parameter)
            second = output.read_bytes()
        self.assertEqual(first, second)
        self.assertIn(b"Gradient distributions", first)
        self.assertIn(parameter.encode(), first)

    def test_gradient_renderer_rejects_uncaptured_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self._corpus(directory)
            result = run_initialization_comparison(
                corpus,
                model_config=TransformerConfig(
                    vocab_size=corpus.vocab_size,
                    block_size=4,
                    embedding_dim=8,
                    head_count=2,
                    layer_count=1,
                ),
                training_config=TrainingConfig(steps=1, batch_size=4),
            )
            with self.assertRaisesRegex(TransformerLabError, "not captured"):
                render_gradient_histograms(
                    Path(directory) / "gradient.svg",
                    result,
                    parameter_name="missing",
                )


if __name__ == "__main__":
    unittest.main()
