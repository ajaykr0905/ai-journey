from __future__ import annotations

import unittest

import torch

from ai_journey.causal_average import CausalAverageError, causal_average_cumsum
from ai_journey.causal_stream import CausalAverageStream


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


if __name__ == "__main__":
    unittest.main()
