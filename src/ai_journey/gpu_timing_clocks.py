"""Three clocks for one Metal kernel launch, and the rule that keeps them apart.

A launch through PyTorch's MPS backend only encodes work on the host; the
GPU runs it later. So "how long did the kernel take" has three answers, and
this lab records all three for every sample side:

- ``mps_event``: ``torch.mps.Event(enable_timing=True)`` recorded just before
  and just after the launch. It times the interval between two points in the
  MPS command stream, so host-side Python cost outside the stream is absent.
- ``host_synchronized``: ``time.perf_counter_ns`` before the launch and after
  the ``torch.mps.synchronize()`` that follows it. It is what a caller waits.
- ``host_dispatch``: ``time.perf_counter_ns`` around the launch call alone,
  with no synchronize between the readings. It is the host cost to encode
  and queue the launch, and it does not wait for the GPU.

The event clock and the host clock belong to different domains. The PyTorch
MPS header says a command-buffer completion handler stores each timed
event's completion time; it does not say which clock that handler reads. So
no event value is ever subtracted from a host value without the explicit
``cross_domain_difference`` record, which names both domains and says what
the difference does and does not mean.

Event times arrive as float milliseconds. They are kept raw beside integer
nanoseconds from ``round(ms * 1e6)``, so a verifier can redo the conversion
and every statistic uses the same nearest-rank rule as the Day 73 and 74 labs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .matmul_metal_mapping import MatmulMetalError
from .memory_access_benchmark import TimingSummary, summarize_timings


class GpuTimingError(MatmulMetalError):
    """Raised when a GPU timing control, sample, fit or report is invalid."""


EVENT = "mps_event"
SYNCHRONIZED = "host_synchronized"
DISPATCH = "host_dispatch"
CLOCKS = (EVENT, SYNCHRONIZED, DISPATCH)
DEVICE_DOMAIN = "mps_event_stream"
HOST_DOMAIN = "host_perf_counter_ns"
CLOCK_DOMAINS = {EVENT: DEVICE_DOMAIN, SYNCHRONIZED: HOST_DOMAIN, DISPATCH: HOST_DOMAIN}
NS_PER_MS = 1_000_000
# One event interval longer than a minute means the run is broken.
MAX_EVENT_MS = 60_000.0
CLOCK_DEFINITIONS = {
    EVENT: (
        "torch.mps.Event(enable_timing=True): start.record() just before the "
        "launch and end.record() just after it, then torch.mps.synchronize() "
        "and start.elapsed_time(end) in ms; samples_ns = round(ms * 1e6)"
    ),
    SYNCHRONIZED: (
        "time.perf_counter_ns() just before the launch and just after the "
        "torch.mps.synchronize() that follows it; the queue is drained before "
        "the first reading"
    ),
    DISPATCH: (
        "time.perf_counter_ns() just before and just after the launch call "
        "alone, with no synchronize between the readings; the queue is drained "
        "before the first reading and after the second"
    ),
}
DOMAIN_DEFINITIONS = {
    DEVICE_DOMAIN: (
        "device-side: the interval between two points in the MPS command "
        "stream; the PyTorch MPS header says a command-buffer completion "
        "handler stores each timed event's completion time and does not name "
        "the clock it reads, so these values are never subtracted from host "
        "values without a cross_domain record"
    ),
    HOST_DOMAIN: (
        "host-side: time.perf_counter_ns on the CPU thread that issued the launch"
    ),
}
CROSS_DOMAIN_MEANING = (
    "cross-domain: the host_synchronized median minus the mps_event median of "
    "the same side, from separate samples on two clocks; an estimate of the "
    "host cost around the GPU work (submit, wait and wake), not a "
    "single-clock interval"
)


def event_ns(milliseconds: object) -> int:
    """Convert one event interval in ms to positive integer nanoseconds.

    The rule is ``round(ms * 1e6)``, so a verifier with the raw value can
    redo it exactly.
    """

    if (
        type(milliseconds) is not float
        or not math.isfinite(milliseconds)
        or not 0.0 < milliseconds <= MAX_EVENT_MS
    ):
        raise GpuTimingError(
            f"an MPS event interval must be a float in (0, {MAX_EVENT_MS}] ms"
        )
    nanoseconds = round(milliseconds * NS_PER_MS)
    if nanoseconds < 1:
        raise GpuTimingError("an MPS event interval rounded to 0 ns")
    return nanoseconds


def require_clock(clock: object) -> str:
    """Return a known clock name, so a typo cannot pick a domain."""

    if clock not in CLOCKS:
        raise GpuTimingError(f"clock must be one of {', '.join(CLOCKS)}")
    return clock


def require_same_domain(first: str, second: str) -> str:
    """Return the shared domain of two clocks; refuse a silent mix."""

    domain = CLOCK_DOMAINS[require_clock(first)]
    if CLOCK_DOMAINS[require_clock(second)] != domain:
        raise GpuTimingError(
            f"{first} and {second} are in different clock domains; use "
            "cross_domain_difference"
        )
    return domain


def cross_domain_difference(host_ns: int, device_ns: int) -> dict[str, object]:
    """Subtract an event median from a host median, labeled as cross-domain."""

    for name, value in (("host_ns", host_ns), ("device_ns", device_ns)):
        if type(value) is not int or value <= 0:
            raise GpuTimingError(f"{name} must be a positive integer")
    return {
        "value_ns": host_ns - device_ns,
        "cross_domain": True,
        "minuend": SYNCHRONIZED,
        "subtrahend": EVENT,
    }


@dataclass(frozen=True)
class ClockSamples:
    """The timed samples of one clock for one side, in run order.

    ``elapsed_ms`` holds the raw event values and exists only for the event
    clock; host clocks are integer nanoseconds from the start.
    """

    clock: str
    samples_ns: tuple[int, ...]
    elapsed_ms: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        require_clock(self.clock)
        if type(self.samples_ns) is not tuple or not self.samples_ns:
            raise GpuTimingError("samples_ns must be a nonempty tuple")
        if self.clock == EVENT:
            raw = self.elapsed_ms
            if type(raw) is not tuple or len(raw) != len(self.samples_ns):
                raise GpuTimingError("event samples need one raw ms value each")
            if tuple(event_ns(value) for value in raw) != self.samples_ns:
                raise GpuTimingError("event samples_ns must be round(ms * 1e6)")
        elif self.elapsed_ms is not None:
            raise GpuTimingError("only the event clock records raw ms values")
        summarize_timings(self.samples_ns)

    @classmethod
    def from_event_ms(cls, elapsed_ms: tuple[float, ...]) -> ClockSamples:
        """Build event samples from raw ``elapsed_time`` values."""

        if type(elapsed_ms) is not tuple:
            raise GpuTimingError("elapsed_ms must be a tuple")
        return cls(EVENT, tuple(event_ns(value) for value in elapsed_ms), elapsed_ms)

    @property
    def domain(self) -> str:
        """Name the clock domain, so a reader cannot mix the two."""

        return CLOCK_DOMAINS[self.clock]

    @property
    def summary(self) -> TimingSummary:
        """Nearest-rank min, quartiles, median, max and IQR of the samples."""

        return summarize_timings(self.samples_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return raw samples beside their statistics, so both can be checked."""

        mapping: dict[str, object] = {
            "clock": self.clock,
            "domain": self.domain,
            "samples_ns": list(self.samples_ns),
        }
        if self.elapsed_ms is not None:
            mapping["elapsed_ms"] = list(self.elapsed_ms)
        mapping["timing"] = self.summary.to_mapping()
        return mapping
