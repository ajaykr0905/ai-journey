"""Name the top bottleneck of one profiled shape and set it against Day 74 ceilings.

This is an inference from host timings and the measured Day 74 ceilings,
not a reading of hardware counters; this host has none that the lab can
read. The trace gives, per side and iteration, the host time of the
dispatch span and of the synchronize span. The GPU work of a side shows
only as the host's wait inside its synchronize span.

The bottleneck is the op with the most self time in the trace. Its kind
comes from its name in the capture layout:

- ``gpu_wait``: a ``<side>/synchronize`` span, where the host waits until the
  queued GPU work has finished;
- ``host_dispatch``: a ``<side>/dispatch`` span or an aten op inside one;
- ``host_other``: iteration or side spans, host time outside both phases.

For each side, the module also turns the median host time into a
host-observed GFLOP/s and compares it with three floors at the Day 74
ceilings: the FMA floor (FLOPs at the FMA ceiling), the compulsory-traffic
floor (A, B and C each moved once at the bandwidth ceiling), and the
no-reuse floor (every multiply-add reading both operands from memory at
the bandwidth ceiling). A GFLOP/s is one FLOP per nanosecond, and a GB/s is
one byte per nanosecond, so each floor is a plain division.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matmul_metal_runtime import SIDES
from .memory_access_tiling import MatmulShape
from .profile_trace_analysis import TraceAnalysis, median_low
from .profile_trace_config import Day74Ceilings
from .profile_trace_events import ParsedTrace, ProfileTraceError
from .profile_trace_layout import DISPATCH, SYNCHRONIZE, phase_region, side_region

FLOAT32_BYTES = 4
GPU_WAIT = "gpu_wait"
HOST_DISPATCH = "host_dispatch"
HOST_OTHER = "host_other"
KINDS = (GPU_WAIT, HOST_DISPATCH, HOST_OTHER)
BASIS = (
    "inference from host-side torch.profiler spans and the measured Day 74 "
    "ceilings of this host; no hardware counter, GPU kernel profiler or CUDA "
    "tool was used"
)


def compulsory_bytes(shape: MatmulShape) -> int:
    """Bytes to read A and B once and write C once, the least traffic possible."""

    return FLOAT32_BYTES * (shape.m * shape.k + shape.k * shape.n + shape.m * shape.n)


def no_reuse_bytes(shape: MatmulShape) -> int:
    """Bytes if every multiply-add read both operands from memory, plus C."""

    return FLOAT32_BYTES * (2 * shape.m * shape.n * shape.k + shape.m * shape.n)


def bottleneck_kind(name: str) -> str:
    """Classify an op name of the capture layout as one of ``KINDS``."""

    for side in SIDES:
        if name == phase_region(side, SYNCHRONIZE):
            return GPU_WAIT
        if name == phase_region(side, DISPATCH):
            return HOST_DISPATCH
    if name.startswith("aten::"):
        return HOST_DISPATCH
    return HOST_OTHER


def _durations(trace: ParsedTrace, name: str) -> tuple[int, ...]:
    values = tuple(
        item.span.duration_ns
        for item in sorted(trace.nested, key=lambda item: item.span.start_ns)
        if item.span.name == name
    )
    if not values:
        raise ProfileTraceError(f"trace holds no {name!r} spans")
    return values


@dataclass(frozen=True)
class SideFigures:
    """The medians of one side's host spans and its rates against the ceilings."""

    side: str
    host_ns: int
    dispatch_ns: int
    synchronize_ns: int
    device_event_ns: int | None
    shape: MatmulShape
    ceilings: Day74Ceilings

    @property
    def gflops_per_second(self) -> float:
        """The host-observed rate: FLOPs over the median side span."""

        return self.shape.flops / self.host_ns

    def to_mapping(self) -> dict[str, object]:
        """Return the figures, rates and floor ratios as a report records them."""

        ceilings = self.ceilings
        bandwidth = ceilings.bandwidth_gb_per_second
        floors = {
            "fma_floor_ns": self.shape.flops / ceilings.fma_gflops_per_second,
            "compulsory_traffic_floor_ns": compulsory_bytes(self.shape) / bandwidth,
            "no_reuse_traffic_floor_ns": no_reuse_bytes(self.shape) / bandwidth,
        }
        return {
            "side": self.side,
            "median_host_ns": self.host_ns,
            "median_dispatch_ns": self.dispatch_ns,
            "median_synchronize_ns": self.synchronize_ns,
            "median_device_event_ns": self.device_event_ns,
            "synchronize_share_of_host": self.synchronize_ns / self.host_ns,
            "host_observed_gflops_per_second": self.gflops_per_second,
            "fraction_of_fma_ceiling": (
                self.gflops_per_second / ceilings.fma_gflops_per_second
            ),
            "fraction_of_torch_matmul_ceiling": (
                self.gflops_per_second / ceilings.torch_matmul_gflops_per_second
            ),
            **floors,
            "host_to_floor_ratio": {
                name: self.host_ns / value for name, value in floors.items()
            },
        }


def side_figures(
    trace: ParsedTrace,
    shape: MatmulShape,
    ceilings: Day74Ceilings,
    device_ns: dict[str, tuple[int, ...]] | None = None,
) -> tuple[SideFigures, ...]:
    """Return the median host figures of every side, in ``SIDES`` order."""

    figures = []
    for side in SIDES:
        host = median_low(_durations(trace, side_region(side)))
        if host <= 0:
            raise ProfileTraceError(f"{side} has no host time")
        events = None if device_ns is None else device_ns[side]
        figures.append(
            SideFigures(
                side=side,
                host_ns=host,
                dispatch_ns=median_low(_durations(trace, phase_region(side, DISPATCH))),
                synchronize_ns=median_low(
                    _durations(trace, phase_region(side, SYNCHRONIZE))
                ),
                device_event_ns=None if events is None else median_low(events),
                shape=shape,
                ceilings=ceilings,
            )
        )
    return tuple(figures)


def bottleneck(
    analysis: TraceAnalysis,
    trace: ParsedTrace,
    shape: MatmulShape,
    ceilings: Day74Ceilings,
    device_ns: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, object]:
    """Name the top op by self time, classify it, and add every side's figures."""

    if not isinstance(analysis, TraceAnalysis) or not analysis.top:
        raise ProfileTraceError("a bottleneck needs an analysis with top ops")
    top = analysis.top[0]
    hosts = sum(item.host_ns for item in analysis.iterations)
    return {
        "basis": BASIS,
        "top_op": top.name,
        "kind": bottleneck_kind(top.name),
        "self_ns": top.self_ns,
        "share_of_iteration_host_time": top.self_ns / hosts,
        "sides": [
            figures.to_mapping()
            for figures in side_figures(trace, shape, ceilings, device_ns)
        ],
    }
