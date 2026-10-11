"""Transformer experiment reports must remain strict, recoverable evidence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

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

    def test_concurrent_writers_publish_one_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            barrier = threading.Barrier(2, timeout=10)
            real_write_text = Path.write_text

            def synchronize_fixed_temporary(target, data, **kwargs):
                written = real_write_text(target, data, **kwargs)
                barrier.wait()
                return written

            with mock.patch.object(
                Path,
                "write_text",
                autospec=True,
                side_effect=synchronize_fixed_temporary,
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            write_experiment_report,
                            path,
                            replace(_result(), completed_steps=step),
                        )
                        for step in (1, 2)
                    ]
                    for future in futures:
                        future.result(timeout=15)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(payload["completed_steps"], (1, 2))
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_report_publication_flushes_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with mock.patch("os.open", wraps=os.open) as opened, mock.patch(
                "os.fsync", wraps=os.fsync
            ) as synced:
                write_experiment_report(path, _result())
            self.assertIn(mock.call(path.parent, os.O_RDONLY), opened.call_args_list)
            self.assertEqual(synced.call_count, 2)

    def test_publication_failure_preserves_prior_report_and_cleans_temporary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("previous evidence\n", encoding="utf-8")
            for operation in ("os.fsync", "os.replace"):
                with self.subTest(operation=operation), mock.patch(
                    operation, side_effect=OSError("publication failed")
                ):
                    with self.assertRaisesRegex(OSError, "publication failed"):
                        write_experiment_report(path, _result())
                self.assertEqual(
                    path.read_text(encoding="utf-8"), "previous evidence\n"
                )
                self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
