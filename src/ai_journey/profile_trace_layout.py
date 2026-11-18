"""The annotation layout every capture writes, and the check that a trace has it.

``ProfilerActivity.CPU`` records only host-side spans, and a kernel that
``torch.mps.compile_shader`` built leaves no op of its own in the trace. So
the capture names the host phases itself with ``record_function``:

- ``profile_trace/iteration`` wraps one profiled iteration.
- ``profile_trace/<side>`` wraps one side inside it: a naive kernel or
  ``torch.matmul``. The sides rotate by one place at each iteration, as in
  the Day 74 benchmark, so no side always runs first.
- ``<side>/dispatch`` wraps the call that queues the work; for
  ``torch.matmul`` the aten ops nest inside it.
- ``<side>/synchronize`` wraps ``torch.mps.synchronize()``, the host's wait
  until the queued GPU work has finished.

The GPU work of a side therefore shows up only as host wait time inside its
synchronize span. ``check_layout`` refuses a trace without exactly this
structure, so the analysis never reads a different capture as this one.
Nothing here imports torch.
"""

from __future__ import annotations

import re

from .matmul_metal_runtime import SIDES, TORCH_MATMUL
from .memory_access_tiling import MatmulShape
from .profile_trace_events import NestedSpan, ParsedTrace, ProfileTraceError

ITERATION = "profile_trace/iteration"
DISPATCH = "dispatch"
SYNCHRONIZE = "synchronize"
PHASES = (DISPATCH, SYNCHRONIZE)
_TRACE_FILE = re.compile(r"trace-[0-9]-[0-9]{1,4}x[0-9]{1,4}x[0-9]{1,4}\.json")


class TraceLayoutError(ProfileTraceError):
    """Raised when a trace lacks the annotation layout the capture writes."""


def side_region(side: str) -> str:
    """Return the span name that wraps one side of one iteration."""

    if side not in SIDES:
        raise TraceLayoutError(f"side must be one of {', '.join(SIDES)}")
    return f"profile_trace/{side}"


def phase_region(side: str, phase: str) -> str:
    """Return the span name of one host phase of one side."""

    side_region(side)
    if phase not in PHASES:
        raise TraceLayoutError(f"phase must be one of {', '.join(PHASES)}")
    return f"{side}/{phase}"


def side_order(iteration: int) -> tuple[str, ...]:
    """Return the sides of one iteration, rotated by one place per iteration."""

    shift = iteration % len(SIDES)
    return SIDES[shift:] + SIDES[:shift]


def trace_file_name(index: int, shape: MatmulShape) -> str:
    """Return the stored file name of one shape's trace, a plain base name."""

    name = f"trace-{index}-{shape.m}x{shape.n}x{shape.k}.json"
    return require_trace_file_name(name)


def require_trace_file_name(name: object) -> str:
    """Accept only the base names the capture writes, so no path can escape."""

    if type(name) is not str or _TRACE_FILE.fullmatch(name) is None:
        raise TraceLayoutError("trace file names must look like trace-0-MxNxK.json")
    return name


def _children(trace: ParsedTrace) -> dict[int, list[tuple[int, NestedSpan]]]:
    children: dict[int, list[tuple[int, NestedSpan]]] = {}
    for index, item in enumerate(trace.nested):
        if item.parent is not None:
            children.setdefault(item.parent, []).append((index, item))
    return children


def check_layout(trace: ParsedTrace, iterations: int) -> None:
    """Require ``iterations`` iterations of the capture layout on one thread.

    Each iteration holds one span per side in the rotated order, and each
    side holds exactly its dispatch span and then its synchronize span. The
    naive kernels' dispatch spans hold no child: the profiler records no op
    for a compiled Metal kernel.
    """

    if not isinstance(trace, ParsedTrace):
        raise TypeError("trace must be a ParsedTrace")
    found = [
        (index, item)
        for index, item in enumerate(trace.nested)
        if item.span.name == ITERATION
    ]
    if len(found) != iterations:
        raise TraceLayoutError(
            f"trace holds {len(found)} {ITERATION} spans; expected {iterations}"
        )
    if len({item.span.thread for _, item in found}) != 1 or any(
        item.parent is not None for _, item in found
    ):
        raise TraceLayoutError("iterations must be top-level spans on one thread")
    found.sort(key=lambda pair: pair[1].span.start_ns)
    children = _children(trace)
    for number, (position, _) in enumerate(found):
        sides = children.get(position, [])
        names = [item.span.name for _, item in sides]
        expected = [side_region(side) for side in side_order(number)]
        if names != expected:
            raise TraceLayoutError(
                f"iteration {number} holds {names}; expected {expected}"
            )
        for side, (side_position, _) in zip(side_order(number), sides):
            phases = children.get(side_position, [])
            phase_names = [item.span.name for _, item in phases]
            wanted = [phase_region(side, phase) for phase in PHASES]
            if phase_names != wanted:
                raise TraceLayoutError(
                    f"iteration {number} side {side} holds {phase_names}; "
                    f"expected {wanted}"
                )
            if phases[1][0] in children:
                raise TraceLayoutError(f"{wanted[1]} must not hold child spans")
            if side != TORCH_MATMUL and phases[0][0] in children:
                raise TraceLayoutError(f"{wanted[0]} must not hold child spans")
