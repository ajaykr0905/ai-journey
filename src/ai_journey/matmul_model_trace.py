"""Lane-by-lane traces of the naive matmul kernel code, in program order.

The access model and the memory_access cross-check derive every address from
the stated index expressions. Here the kernel functions themselves run, one
lane of one warp at a time, with buffers that record each index in the order
the code reads or writes it. The ``i``-th load of A or B by a lane belongs to
loop step ``i``. ``trace_matches_model`` requires each lane's loads, in that
order, and its one store to equal what the access model states, so a kernel
whose indexing or loop order drifts from the model fails.

The kernels run outside the simulator here because the simulator's buffers
count accesses but do not keep their order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .kernel_launch_config import Index3
from .kernel_launch_simulator import ThreadContext
from .matmul_model_access import (
    ELEMENT_BYTES,
    MatmulLaunch,
    WarpRef,
    lane_elements,
    warp_requests,
)
from .matmul_model_kernels import KERNELS
from .memory_access_pattern import WARP_SIZE


class TraceBuffer:
    """A stand-in device buffer that records each index in program order.

    A load outside the values returns 0.0, as a simulated out-of-bounds read
    does, so a drifted kernel still runs to the end of its trace.
    """

    __slots__ = ("_values", "loads", "stores")

    def __init__(self, values: Sequence[float]) -> None:
        self._values = tuple(values)
        self.loads: list[int] = []
        self.stores: list[tuple[int, float]] = []

    def __getitem__(self, index: int) -> float:
        if type(index) is not int:
            raise TypeError("trace buffer index must be an integer")
        self.loads.append(index)
        return self._values[index] if 0 <= index < len(self._values) else 0.0

    def __setitem__(self, index: int, value: float) -> None:
        if type(index) is not int:
            raise TypeError("trace buffer index must be an integer")
        self.stores.append((index, value))


@dataclass(frozen=True)
class LaneTrace:
    """The elements that one lane loaded from A and B and stored to C."""

    thread: Index3
    a_loads: tuple[int, ...]
    b_loads: tuple[int, ...]
    c_stores: tuple[tuple[int, float], ...]


def trace_warp(
    launch: MatmulLaunch, ref: WarpRef, a: Sequence[float], b: Sequence[float]
) -> tuple[LaneTrace, ...]:
    """Run the mapping's kernel code for each lane of one warp and record it."""

    lane_elements(launch, ref)
    config = launch.config
    shape = launch.shape
    kernel = KERNELS[launch.mapping]
    first = config.thread_id(ref.block, Index3(0)) + ref.warp * WARP_SIZE
    traces = []
    for lane in range(WARP_SIZE):
        block_idx, thread_idx = config.coordinates(first + lane)
        a_buffer, b_buffer, c_buffer = TraceBuffer(a), TraceBuffer(b), TraceBuffer(())
        kernel(
            ThreadContext(block_idx, thread_idx, config.block, config.grid),
            a_buffer,
            b_buffer,
            c_buffer,
            shape.m,
            shape.n,
            shape.k,
        )
        traces.append(
            LaneTrace(
                thread=thread_idx,
                a_loads=tuple(a_buffer.loads),
                b_loads=tuple(b_buffer.loads),
                c_stores=tuple(c_buffer.stores),
            )
        )
    return tuple(traces)


def trace_matches_model(
    launch: MatmulLaunch,
    ref: WarpRef,
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> bool:
    """Return whether the kernel code loads and stores what the model states.

    ``c`` is the reference output. Each lane that the guard keeps must load
    the model's A and B elements in step order and store its element of
    ``c`` once; every other lane must touch nothing.
    """

    model = warp_requests(launch, ref)
    for lane, trace in enumerate(trace_warp(launch, ref, a, b)):
        address = None if model is None else model.c_store.lane_addresses[lane]
        if address is None:
            expected: tuple[tuple[object, ...], ...] = ((), (), ())
        else:
            index = address // ELEMENT_BYTES
            expected = (
                tuple(
                    load.lane_addresses[lane] // ELEMENT_BYTES for load in model.a_loads
                ),
                tuple(
                    load.lane_addresses[lane] // ELEMENT_BYTES for load in model.b_loads
                ),
                ((index, c[index]),),
            )
        if (trace.a_loads, trace.b_loads, trace.c_stores) != expected:
            return False
    return True
