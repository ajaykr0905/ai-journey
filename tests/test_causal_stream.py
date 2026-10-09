from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from ai_journey.causal_average import CausalAverageError, causal_average_cumsum
from ai_journey.causal_stream import (
    CausalAverageStream,
    StreamSnapshot,
    build_snapshot_payload,
    load_snapshot,
    parse_snapshot_payload,
    save_snapshot,
)


class CausalAverageStreamTests(unittest.TestCase):
    def test_arbitrary_chunking_matches_one_shot_causal_averages(self) -> None:
        values = torch.randn(11, 4, generator=torch.Generator().manual_seed(358))
        stream = CausalAverageStream(4)
        output = torch.cat(
            [
                stream.update(values[:2]),
                stream.update(values[2:7]),
                stream.update(values[7:]),
            ]
        )
        torch.testing.assert_close(output, causal_average_cumsum(values))
        self.assertEqual(stream.count, len(values))

    def test_single_token_updates_match_each_prefix(self) -> None:
        values = torch.arange(15, dtype=torch.float64).reshape(5, 3)
        stream = CausalAverageStream(3, dtype=torch.float64)
        output = torch.cat([stream.update(row.unsqueeze(0)) for row in values])
        torch.testing.assert_close(output, causal_average_cumsum(values))

    def test_rejects_misaligned_chunks_without_advancing_count(self) -> None:
        stream = CausalAverageStream(3)
        for chunk, message in (
            (torch.ones(1, 4), "feature size"),
            (torch.ones(2, 3, dtype=torch.float64), "dtype and device"),
            (torch.ones(1, 2, 3), "shape"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(CausalAverageError, message):
                    stream.update(chunk)
                self.assertEqual(stream.count, 0)

    def test_snapshot_restores_exact_continuation(self) -> None:
        values = torch.randn(9, 3, generator=torch.Generator().manual_seed(359))
        stream = CausalAverageStream(3)
        stream.update(values[:4])
        snapshot = stream.snapshot()
        expected = stream.update(values[4:])
        stream.restore(snapshot)
        actual = stream.update(values[4:])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_snapshot_is_returned_and_restored_by_value(self) -> None:
        stream = CausalAverageStream(2)
        stream.update(torch.tensor([[2.0, 4.0]]))
        snapshot = stream.snapshot()
        snapshot.total.zero_()
        self.assertTrue(torch.equal(stream.snapshot().total, torch.tensor([2.0, 4.0])))
        stream.restore(StreamSnapshot(1, torch.tensor([3.0, 5.0])))
        replacement = stream.snapshot()
        replacement.total.add_(100)
        self.assertTrue(torch.equal(stream.snapshot().total, torch.tensor([3.0, 5.0])))

    def test_invalid_snapshot_cannot_mutate_live_state(self) -> None:
        stream = CausalAverageStream(2)
        stream.update(torch.tensor([[1.0, 2.0]]))
        before = stream.snapshot()
        invalid = StreamSnapshot(4, torch.tensor([float("nan"), 0.0]))
        with self.assertRaisesRegex(CausalAverageError, "finite"):
            stream.restore(invalid)
        after = stream.snapshot()
        self.assertEqual(after.count, before.count)
        self.assertTrue(torch.equal(after.total, before.total))

    def test_serialized_snapshot_round_trips_exactly(self) -> None:
        snapshot = StreamSnapshot(3, torch.tensor([1.5, -2.25], dtype=torch.float64))
        payload = build_snapshot_payload(snapshot)
        restored = parse_snapshot_payload(payload)
        self.assertEqual(restored.count, snapshot.count)
        self.assertEqual(restored.total.dtype, snapshot.total.dtype)
        self.assertTrue(torch.equal(restored.total, snapshot.total))

    def test_snapshot_payload_detects_tampering(self) -> None:
        payload = build_snapshot_payload(StreamSnapshot(2, torch.tensor([3.0, 4.0])))
        payload["total"][0] = 999.0
        with self.assertRaisesRegex(CausalAverageError, "fingerprint mismatch"):
            parse_snapshot_payload(payload)

    def test_snapshot_payload_rejects_unknown_fields(self) -> None:
        payload = build_snapshot_payload(StreamSnapshot(0, torch.zeros(2)))
        payload["unexpected"] = True
        with self.assertRaisesRegex(CausalAverageError, "fields"):
            parse_snapshot_payload(payload)

    def test_snapshot_file_round_trips_and_leaves_no_temporary(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "nested", "stream.json")
            snapshot = StreamSnapshot(7, torch.tensor([1.25, -3.5]))
            save_snapshot(path, snapshot)
            restored = load_snapshot(path)
            self.assertEqual(restored.count, snapshot.count)
            self.assertTrue(torch.equal(restored.total, snapshot.total))
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_failed_replacement_preserves_existing_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "stream.json")
            path.write_text("preserve me", encoding="utf-8")
            with patch(
                "ai_journey.causal_stream.os.replace", side_effect=OSError("boom")
            ):
                with self.assertRaisesRegex(OSError, "boom"):
                    save_snapshot(path, StreamSnapshot(1, torch.ones(2)))
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_loader_rejects_truncated_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "stream.json")
            path.write_text('{"schema":', encoding="utf-8")
            with self.assertRaisesRegex(CausalAverageError, "cannot read"):
                load_snapshot(path)


if __name__ == "__main__":
    unittest.main()
