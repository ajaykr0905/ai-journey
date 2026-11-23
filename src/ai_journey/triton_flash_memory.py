"""A declared memory and traffic model of naive and flash attention, per head.

This is a model of allocated intermediates and of bytes moved. It is not a
measured GPU footprint: no GPU ran, and an allocator, a workspace, a cache or
a fused library kernel can change the real numbers. The model counts float32
values of one head of one batch element, with sequence length N and head dim d.

Memory rule:

* naive attention allocates Q, K, V and O (``4 N d`` values) and materializes
  the ``N x N`` score matrix S and the ``N x N`` probability matrix P, so it
  holds ``2 N N + 4 N d`` values: ``2 * N * N * 4`` bytes of intermediates;
* flash attention allocates the same Q, K, V and O and only the per-row
  running max m and running sum l, ``2 N`` values. Its tiles live in registers
  or shared memory for one loop step, so the model counts no N x N buffer.

Traffic rule, for memory outside the chip, with no cache:

* naive attention reads Q and K, writes S, reads S, writes P, reads P and V
  and writes O: ``4 N N + 4 N d`` values;
* flash attention reads Q once, reads K and V once for each of the
  ``cdiv(N, BLOCK_M)`` query blocks, writes O and writes m and l:
  ``2 N d + 2 N d cdiv(N, BLOCK_M) + 2 N`` values, with ``BLOCK_M = 128`` of
  the first declared config.

The memory gap grows as N / d, because only naive attention holds N x N
values. The traffic gap is near ``4 N N / (2 N N d / BLOCK_M)``, which is
``2 BLOCK_M / d``, because flash attention reads K and V again for each query
block. The FlashAttention paper states this trade as ``O(N^2 d^2 / M)``
accesses for an on-chip memory of M values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernel_metal_geometry import ceil_div
from .triton_flash_kernel import FLASH_CONFIGS, FLOAT32_BYTES, _integer

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192)
MODEL_HEAD_DIMS = (64, 128)
TRAFFIC_BLOCK_M = FLASH_CONFIGS[0].block_m
MAX_MODEL_SEQUENCE = 1 << 20
MEMORY_RULE = (
    "per head, float32: naive = 2 * N * N * 4 bytes of S and P plus 4 * N * d * 4 "
    "bytes of Q, K, V and O; flash = 4 * N * d * 4 bytes of Q, K, V and O plus "
    "2 * N * 4 bytes of m and l"
)
TRAFFIC_RULE = (
    "per head, float32, no cache: naive = (4 * N * N + 4 * N * d) * 4 bytes; "
    "flash = (2 * N * d + 2 * N * d * cdiv(N, 128) + 2 * N) * 4 bytes"
)
STATEMENT = (
    "This is a model of allocated intermediates and of bytes moved, from a "
    "declared rule. It is not a measured GPU footprint, and no GPU ran."
)


@dataclass(frozen=True)
class MemoryRow:
    """The modeled bytes of one head for one sequence length and head dim."""

    sequence: int
    head_dim: int

    def __post_init__(self) -> None:
        _integer(self.sequence, "sequence", 1, MAX_MODEL_SEQUENCE)
        _integer(self.head_dim, "head_dim", 1, 1024)

    @property
    def io_bytes(self) -> int:
        """Q, K, V and O: both forms allocate them."""

        return 4 * self.sequence * self.head_dim * FLOAT32_BYTES

    @property
    def naive_intermediate_bytes(self) -> int:
        return 2 * self.sequence * self.sequence * FLOAT32_BYTES

    @property
    def flash_intermediate_bytes(self) -> int:
        return 2 * self.sequence * FLOAT32_BYTES

    @property
    def naive_bytes(self) -> int:
        return self.naive_intermediate_bytes + self.io_bytes

    @property
    def flash_bytes(self) -> int:
        return self.flash_intermediate_bytes + self.io_bytes

    @property
    def naive_traffic_bytes(self) -> int:
        n, d = self.sequence, self.head_dim
        return (4 * n * n + 4 * n * d) * FLOAT32_BYTES

    @property
    def flash_traffic_bytes(self) -> int:
        n, d = self.sequence, self.head_dim
        blocks = ceil_div(n, TRAFFIC_BLOCK_M)
        return (2 * n * d + 2 * n * d * blocks + 2 * n) * FLOAT32_BYTES

    def to_mapping(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "head_dim": self.head_dim,
            "io_bytes": self.io_bytes,
            "naive_intermediate_bytes": self.naive_intermediate_bytes,
            "naive_bytes": self.naive_bytes,
            "flash_intermediate_bytes": self.flash_intermediate_bytes,
            "flash_bytes": self.flash_bytes,
            "memory_ratio": self.naive_bytes / self.flash_bytes,
            "naive_traffic_bytes": self.naive_traffic_bytes,
            "flash_traffic_bytes": self.flash_traffic_bytes,
            "traffic_ratio": self.naive_traffic_bytes / self.flash_traffic_bytes,
        }


MEMORY_TABLE = tuple(
    MemoryRow(sequence, dim) for dim in MODEL_HEAD_DIMS for sequence in SEQUENCE_LENGTHS
)


def memory_mapping() -> dict[str, object]:
    """Return the rules, the boundary statement and every row of the table."""

    return {
        "unit": "bytes per head, float32",
        "memory_rule": MEMORY_RULE,
        "traffic_rule": TRAFFIC_RULE,
        "traffic_block_m": TRAFFIC_BLOCK_M,
        "measured": False,
        "statement": STATEMENT,
        "rows": [row.to_mapping() for row in MEMORY_TABLE],
    }


def mebibytes(count: int) -> str:
    """Format a byte count in MiB with two decimals, for the command line."""

    return f"{count / (1 << 20):.2f}"
