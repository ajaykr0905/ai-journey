"""The memory of each side: allocator counters beside the bytes of the design.

For each timing point the runtime runs each side alone and reads the two
MPS counters at four moments: before the side allocates, after it
allocates, the largest value that a second thread polled while the call
ran, and after the call. This module checks those readings for form,
derives three differences from them, and sets beside them the bytes that
each side must allocate by its construction:

- N0: Q, K, V and O, the N x N score buffer of 4 L * L bytes for each
  sequence, and the index buffers of its launches.
- F1: Q, K, V and O and the index buffers. Its tiles live in threadgroup
  memory, which the counters do not count.
- The PyTorch call: Q, K and V, and the O that the call returns. The
  scratch that it needs is not known by construction, because PyTorch names
  no kernel.

``COUNTERS_MEASURE`` and ``COUNTERS_CANNOT_SHOW`` state what the readings
mean and where they stop.
"""

from __future__ import annotations

from collections.abc import Mapping

from .metal_flash_plan import SIDES, TORCH_SDPA
from .metal_flash_reference import FlashLayout
from .metal_flash_source import F1, FLOAT32_BYTES, N0, UINT32_BYTES, MetalFlashError
from .metal_flash_runtime import COUNTERS, READINGS

MAX_BYTES = 1 << 40
MAX_POLLS = 1 << 40
COUNTERS_MEASURE = (
    "torch.mps.current_allocated_memory() is the bytes of live tensors in the "
    "PyTorch MPS allocator, rounded up to its blocks; "
    "torch.mps.driver_allocated_memory() is the bytes that the Metal driver "
    "holds for this process, with the allocator cache and the buffers of the "
    "MPS frameworks; each side runs alone after torch.mps.empty_cache(), and a "
    "second thread polls both counters while its call runs"
)
COUNTERS_CANNOT_SHOW = (
    "a buffer that lives and dies between two polls; threadgroup memory, "
    "registers and other on-chip storage; memory that Metal or the operating "
    "system holds outside these two counters; which kernel PyTorch ran; the "
    "driver counter grows in large heaps, so it is coarse"
)
CONSTRUCTION = {
    N0: (
        "Q, K, V and O, 16 * 64 bytes for each buffer row; the score buffer, 4 "
        "L * L bytes for each sequence; starts, lengths and score starts, 12 "
        "bytes for each sequence; the sequence and row of each query group, 8 "
        "bytes for each token"
    ),
    F1: (
        "Q, K, V and O, 16 * 64 bytes for each buffer row; starts and lengths, "
        "8 bytes for each sequence; the sequence and block of each group, 8 "
        "bytes for each block of 32 queries; the K and V tiles are threadgroup "
        "memory"
    ),
    TORCH_SDPA: (
        "Q, K and V, and the O that the call returns, 16 * 64 bytes for each "
        "buffer row; any scratch of the call is not known by construction"
    ),
}


def construction_bytes(side: str, layout: FlashLayout) -> dict[str, int | None]:
    """Return the bytes that one side allocates by its design, by kind."""

    if not isinstance(layout, FlashLayout):
        raise TypeError("layout must be a FlashLayout")
    tensors = 4 * FLOAT32_BYTES * layout.buffer_elements
    sequences = layout.sequences
    if side == N0:
        return {
            "tensors": tensors,
            "scratch": FLOAT32_BYTES * layout.score_elements,
            "index": UINT32_BYTES * (3 * sequences + 2 * layout.tokens),
        }
    if side == F1:
        return {
            "tensors": tensors,
            "scratch": 0,
            "index": UINT32_BYTES * (2 * sequences + 2 * layout.blocks()),
        }
    if side == TORCH_SDPA:
        return {"tensors": tensors, "scratch": None, "index": 0}
    raise MetalFlashError(f"{side!r} is not a timed side")


def _counters(value: object, label: str) -> dict[str, int]:
    if (
        type(value) is not dict
        or set(value) != set(COUNTERS)
        or not all(
            type(value[name]) is int and 0 <= value[name] <= MAX_BYTES
            for name in COUNTERS
        )
    ):
        raise MetalFlashError(f"{label} must hold two byte counts")
    return dict(value)


def side_memory(
    side: str, layout: FlashLayout, readings: Mapping[str, object]
) -> dict[str, object]:
    """Return one side's readings with the derived differences and its design."""

    if type(readings) is not dict or set(readings) != {*READINGS, "polls"}:
        raise MetalFlashError(f"the memory readings of {side} do not match the form")
    values = {name: _counters(readings[name], f"{side} {name}") for name in READINGS}
    polls = readings["polls"]
    if type(polls) is not int or not 1 <= polls <= MAX_POLLS:
        raise MetalFlashError(f"the poll count of {side} must be a positive integer")
    built = construction_bytes(side, layout)
    known = sum(value for value in built.values() if value is not None)
    allocated = (
        values["after_allocation"]["current_bytes"] - values["before"]["current_bytes"]
    )
    return {
        "readings": {**values, "polls": polls},
        "construction": CONSTRUCTION[side],
        "construction_bytes": built,
        "known_construction_bytes": known,
        "allocated_current_bytes": allocated,
        "peak_current_over_allocation_bytes": values["peak_during_call"][
            "current_bytes"
        ]
        - values["after_allocation"]["current_bytes"],
        "peak_driver_over_before_bytes": values["peak_during_call"]["driver_bytes"]
        - values["before"]["driver_bytes"],
    }


def point_memory(layout: FlashLayout, readings: object) -> dict[str, object]:
    """Return the memory record of every side of one point, in plan order."""

    if type(readings) is not dict or set(readings) != set(SIDES):
        raise MetalFlashError("memory readings must cover the three sides")
    return {side: side_memory(side, layout, readings[side]) for side in SIDES}
