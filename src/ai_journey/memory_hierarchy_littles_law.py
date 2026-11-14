"""Little's law over measured rates: what a stated latency implies is in flight.

Little's law says the average number of requests in flight is their
completion rate times how long each one takes. Multiplying a rate measured on
this host by a latency the config states gives the concurrency that rate
would need if every access waited that long: bytes in flight for triad's
bandwidth at the largest working set, and source elements in flight for each
access pattern. Fields keep three kinds of number apart: ``measured_`` rates
come from this host's median samples, ``stated_latency_ns`` is a config
input that this lab never measures, and ``derived_`` values are their
product, an implication rather than an observation. Nothing here counts the
requests that were actually outstanding.
"""

from __future__ import annotations

from .memory_hierarchy_access import PATTERN_NAMES, AccessMeasurement
from .memory_hierarchy_config import LatencyControls
from .memory_hierarchy_stream import MemoryHierarchyError
from .memory_hierarchy_sweep import StreamMeasurement
from .memory_hierarchy_timing import per_nanosecond

LAW = "in_flight = measured_rate_per_ns * stated_latency_ns"


def littles_law_section(
    latency: LatencyControls,
    stream: tuple[StreamMeasurement, ...],
    access: tuple[AccessMeasurement, ...],
) -> dict[str, object]:
    """Derive in-flight bytes and elements from measured rates and a stated latency.

    The bandwidth row uses the same triad measurement as the roofline's
    placement ceiling, so both sections describe one observed rate.
    """

    if not isinstance(latency, LatencyControls):
        raise TypeError("latency must be LatencyControls")
    if type(stream) is not tuple or any(
        not isinstance(item, StreamMeasurement) for item in stream
    ):
        raise TypeError("stream must be a tuple of StreamMeasurement")
    if type(access) is not tuple or any(
        not isinstance(item, AccessMeasurement) for item in access
    ):
        raise TypeError("access must be a tuple of AccessMeasurement")
    if tuple(item.pattern for item in access) != PATTERN_NAMES:
        raise MemoryHierarchyError("access rows must cover every pattern in order")
    triads = [item for item in stream if item.kernel == "triad"]
    if not triads:
        raise MemoryHierarchyError("the bandwidth row needs triad measurements")
    triad = max(triads, key=lambda item: item.elements)
    stated = latency.latency_ns
    bytes_per_ns = per_nanosecond(
        triad.bytes_per_sample, triad.timing.summary.median_ns
    )
    return {
        "law": LAW,
        "stated_latency_ns": stated,
        "latency_note": latency.latency_note,
        "latency_measured": False,
        "bandwidth": {
            "source": "stream_triad_largest_working_set",
            "elements": triad.elements,
            "measured_bytes_per_ns": bytes_per_ns,
            "derived_bytes_in_flight": bytes_per_ns * stated,
        },
        "access_patterns": [
            {
                "pattern": item.pattern,
                "measured_elements_per_ns": item.median_elements_per_ns,
                "derived_elements_in_flight": item.median_elements_per_ns * stated,
            }
            for item in access
        ],
    }
