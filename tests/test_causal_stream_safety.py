import unittest
import torch

from ai_journey.causal_average import CausalAverageError
from ai_journey.causal_stream import (
    CausalAverageStream,
    StreamSnapshot,
    build_snapshot_payload,
)


class StreamSafetyTests(unittest.TestCase):
    def test_finite_input_overflow_cannot_poison_restart_state(self):
        stream = CausalAverageStream(1)
        before = stream.snapshot()
        with self.assertRaisesRegex(CausalAverageError, "finite"):
            stream.update(torch.tensor([[3e38], [3e38]]))
        self.assertEqual(stream.count, before.count)
        self.assertTrue(torch.equal(stream.snapshot().total, before.total))
        self.assertEqual(float(stream.update(torch.tensor([[2.0]]))[0, 0]), 2.0)

    def test_zero_count_nonzero_total_is_unreachable_and_transactionally_rejected(self):
        stream = CausalAverageStream(1)
        stream.update(torch.ones(1, 1))
        bad = StreamSnapshot(0, torch.ones(1))
        with self.assertRaises(CausalAverageError):
            stream.restore(bad)
        self.assertEqual(stream.count, 1)
        with self.assertRaises(CausalAverageError):
            build_snapshot_payload(bad)
        with self.assertRaises(CausalAverageError):
            build_snapshot_payload(StreamSnapshot(0, torch.empty(0)))
