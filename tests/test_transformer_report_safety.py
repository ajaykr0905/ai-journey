"""Transformer experiment reports must remain strict, recoverable evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from ai_journey.transformer_lab import (
    ExperimentResult,
    TrainingConfig,
    TransformerConfig,
    write_experiment_report,
)


def _result() -> ExperimentResult:
    return ExperimentResult(
        model_config=TransformerConfig(
            vocab_size=5,
            block_size=4,
            embedding_dim=8,
            head_count=2,
            layer_count=1,
        ),
        training_config=TrainingConfig(steps=1, batch_size=1),
        corpus_fingerprint="a" * 64,
        codec_fingerprint="b" * 64,
        model_fingerprint="c" * 64,
        parameter_count=100,
        completed_steps=1,
        initial_train_nll=2.0,
        final_train_nll=1.5,
        initial_validation_nll=2.1,
        final_validation_nll=1.6,
        trace=(),
    )


class TransformerReportSafetyTests(unittest.TestCase):
    def test_nonfinite_report_preserves_previous_evidence_before_io(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "report.json"
            path.write_text("previous evidence\n", encoding="utf-8")
            invalid = replace(_result(), final_validation_nll=float("nan"))

            with self.assertRaisesRegex(ValueError, "JSON"):
                write_experiment_report(path, invalid)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous evidence\n")
            self.assertEqual(list(root.iterdir()), [path])

            nested = root / "not-created" / "report.json"
            with self.assertRaisesRegex(ValueError, "JSON"):
                write_experiment_report(nested, invalid)
            self.assertFalse(nested.parent.exists())


if __name__ == "__main__":
    unittest.main()
