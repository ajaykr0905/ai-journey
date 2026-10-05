from __future__ import annotations

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_journey.ablation_protocol import write_ablation_report


class AblationReportPublicationTests(unittest.TestCase):
    def test_concurrent_writers_publish_their_own_complete_reports(self) -> None:
        # Pause both writers after serialization, before publication. Neither may
        # rename the other writer's temporary or report a spurious missing file.
        barrier = threading.Barrier(2, timeout=10)
        original_replace = Path.replace

        def synchronized_replace(source: Path, destination: Path) -> Path:
            barrier.wait()
            return original_replace(source, destination)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with (
                patch(
                    "ai_journey.ablation_protocol.build_ablation_report",
                    side_effect=lambda value: value,
                ),
                patch.object(Path, "replace", synchronized_replace),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [
                    executor.submit(write_ablation_report, output, {"writer": writer})
                    for writer in (1, 2)
                ]
                for future in futures:
                    future.result(timeout=15)
            self.assertIn(
                json.loads(output.read_text()), ({"writer": 1}, {"writer": 2})
            )
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_publication_failure_preserves_previous_report_and_cleans_temporary(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_bytes(b"previous evidence")
            with (
                patch(
                    "ai_journey.ablation_protocol.build_ablation_report",
                    return_value={"writer": 1},
                ),
                patch.object(
                    Path, "replace", side_effect=OSError("publication failed")
                ),
                self.assertRaisesRegex(OSError, "publication failed"),
            ):
                write_ablation_report(output, object())
            self.assertEqual(output.read_bytes(), b"previous evidence")
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_existing_legacy_temporary_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            legacy_temporary = output.with_name(".report.json.tmp")
            legacy_temporary.write_bytes(b"another writer's file")
            with patch(
                "ai_journey.ablation_protocol.build_ablation_report",
                return_value={"writer": 1},
            ):
                write_ablation_report(output, object())
            self.assertEqual(legacy_temporary.read_bytes(), b"another writer's file")
            self.assertEqual(json.loads(output.read_text()), {"writer": 1})

    def test_flush_failure_does_not_publish_or_leave_owned_temporary(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_bytes(b"previous evidence")
            with (
                patch(
                    "ai_journey.ablation_protocol.build_ablation_report",
                    return_value={"writer": 1},
                ),
                patch(
                    "ai_journey.ablation_protocol.os.fsync",
                    side_effect=OSError("flush failed"),
                ),
                self.assertRaisesRegex(OSError, "flush failed"),
            ):
                write_ablation_report(output, object())
            self.assertEqual(output.read_bytes(), b"previous evidence")
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_nonfinite_json_is_rejected_before_creating_output_directory(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "report.json"
            with (
                patch(
                    "ai_journey.ablation_protocol.build_ablation_report",
                    return_value={"loss": float("nan")},
                ),
                self.assertRaises(ValueError),
            ):
                write_ablation_report(output, object())
            self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
