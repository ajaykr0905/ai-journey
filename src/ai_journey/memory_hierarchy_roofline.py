"""A measured CPU roofline: matmul compute ceiling, triad bandwidth ceilings.

Every ceiling is a median rate measured on this host. The compute ceiling is
the matmul's FLOP rate. The bandwidth ceiling used for placement is triad's
accounted-byte rate at the largest sweep point, the working set furthest
beyond any cache the sweep could fill; a second ceiling records triad at the
sweep's peak point, so the report also shows the ridge the in-cache rate
would give. ``roofline.RooflineHardware`` computes each ridge point and
``roofline.analyze_roofline`` places every kernel measured at the largest
point that performs FLOPs (scale, add, triad) and the matmul at its
arithmetic intensity. Triad and the matmul sit on the ceilings they define
by construction; the placement shows where the other kernels land.

The ceilings come from NumPy calls: the matmul may run several BLAS threads
while the STREAM kernels run on one, so the ridge describes this lab's NumPy
workloads on this host, not the processor's peak, and nothing about a GPU.
"""

from __future__ import annotations

from .memory_access_roofline import roofline_estimate_mapping
from .memory_hierarchy_matmul import MatmulMeasurement
from .memory_hierarchy_stream import MemoryHierarchyError, element_bytes
from .memory_hierarchy_sweep import StreamMeasurement
from .memory_hierarchy_timing import per_second
from .roofline import KernelWorkload, RooflineHardware, analyze_roofline

CEILING_STATISTIC = "median"
BANDWIDTH_KERNEL = "triad"


def _bytes_per_second(item: StreamMeasurement) -> float:
    return per_second(item.bytes_per_sample, item.timing.summary.median_ns)


def _bandwidth_ceiling(
    name: str, item: StreamMeasurement, peak_flops_per_second: float
) -> dict[str, object]:
    hardware = RooflineHardware(
        peak_flops_per_second=peak_flops_per_second,
        memory_bandwidth_bytes_per_second=_bytes_per_second(item),
    )
    return {
        "name": name,
        "kernel": item.kernel,
        "elements": item.elements,
        "working_set_bytes": item.working_set_bytes,
        "bytes_per_second": hardware.memory_bandwidth_bytes_per_second,
        "ridge_point_flops_per_byte": hardware.ridge_point,
    }


def stream_workload(item: StreamMeasurement) -> KernelWorkload:
    """Express one kernel call's FLOPs and accounted bytes as a roofline workload.

    Reads and writes stay separate so the placement records both, exactly as
    the STREAM accounting counted them.
    """

    if not isinstance(item, StreamMeasurement):
        raise TypeError("item must be a StreamMeasurement")
    spec = item.spec
    width = element_bytes(item.dtype)
    return KernelWorkload(
        kind=item.kernel,
        flops=spec.flops_per_call(item.elements),
        bytes_read=spec.reads_per_element * item.elements * width,
        bytes_written=spec.writes_per_element * item.elements * width,
    )


def _placement(
    source: str,
    workload: KernelWorkload,
    hardware: RooflineHardware,
    measured_flops_per_second: float,
) -> dict[str, object]:
    estimate = analyze_roofline(workload, hardware)
    return {
        "source": source,
        **roofline_estimate_mapping(estimate),
        "measured_flops_per_second": measured_flops_per_second,
        "fraction_of_attainable": measured_flops_per_second
        / estimate.attainable_flops_per_second,
    }


def measured_roofline(
    stream: tuple[StreamMeasurement, ...], matmul: MatmulMeasurement
) -> dict[str, object]:
    """Build the roofline section: ceilings, ridge points and kernel placements.

    Medians on both axes keep the ceilings and placements comparable; the
    best samples stay in each measurement for anyone who wants peak rates.
    """

    if type(stream) is not tuple or any(
        not isinstance(item, StreamMeasurement) for item in stream
    ):
        raise TypeError("stream must be a tuple of StreamMeasurement")
    if not isinstance(matmul, MatmulMeasurement):
        raise TypeError("matmul must be a MatmulMeasurement")
    triads = [item for item in stream if item.kernel == BANDWIDTH_KERNEL]
    if not triads:
        raise MemoryHierarchyError("the bandwidth ceiling needs triad measurements")
    if {item.dtype for item in stream} != {matmul.dtype}:
        raise MemoryHierarchyError("ceilings must come from one dtype")
    largest = max(triads, key=lambda item: item.elements)
    peak_point = max(triads, key=lambda item: item.median_gb_per_second)
    workload = matmul.workload
    peak_flops = per_second(workload.flops, matmul.timing.summary.median_ns)
    hardware = RooflineHardware(
        peak_flops_per_second=peak_flops,
        memory_bandwidth_bytes_per_second=_bytes_per_second(largest),
    )
    placements = [
        _placement(
            "stream",
            stream_workload(item),
            hardware,
            per_second(item.flops_per_sample, item.timing.summary.median_ns),
        )
        for item in stream
        if item.elements == largest.elements and item.spec.flops_per_element > 0
    ]
    placements.append(_placement("matmul", workload, hardware, peak_flops))
    return {
        "ceiling_statistic": CEILING_STATISTIC,
        "compute_ceiling": {
            "source": "matmul",
            "dtype": matmul.dtype,
            "m": matmul.m,
            "n": matmul.n,
            "k": matmul.k,
            "flops_per_second": hardware.peak_flops_per_second,
        },
        "bandwidth_ceilings": [
            _bandwidth_ceiling("largest_working_set", largest, peak_flops),
            _bandwidth_ceiling("peak_sweep_point", peak_point, peak_flops),
        ],
        "ridge_point_flops_per_byte": hardware.ridge_point,
        "placement_elements": largest.elements,
        "placements": placements,
    }
