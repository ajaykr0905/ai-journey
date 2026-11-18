"""Take the one-call timing input from a sealed gpu-timing report too.

The Day 75 gpu-timing lab times each launch alone and reads one clock per
sample, so its medians are the one-call times that the bottleneck rule
needs, and no event record sits inside a host interval:

* ``median_gpu_ns`` is the ``mps_event`` median of the side;
* ``median_host_ns`` is the ``host_synchronized`` median;
* ``median_dispatch_ns`` is the ``host_dispatch`` median.

A report passes the gpu-timing verifier, and a file also its canonical-bytes
check, before the rule reads a number from it. Its threadgroup size and
float32 matmul precision must equal those of the Day 74 matmul-metal
report, because other launches or another precision time other work. The
producer names the report digest, so the sealed bottleneck report leads
back to the timing report. A file of any other kind goes to the declared
schema of ``matmul_bottleneck_timing``. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .gpu_timing_clocks import DISPATCH, EVENT, SYNCHRONIZED, GpuTimingError
from .gpu_timing_io import load_report as load_gpu_timing_report
from .gpu_timing_report import REPORT_DIGEST_FIELD, REPORT_KIND
from .gpu_timing_verify import GpuTimingVerificationError, verify_report
from .kernel_metal_report import ENVIRONMENT_FIELDS
from .matmul_bottleneck_config import (
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
)
from .matmul_bottleneck_sources import Day74Sources
from .matmul_bottleneck_timing import TimingEntry, TimingInput, load_timing
from .matmul_metal_mapping import matmul_shape
from .matmul_metal_report import MATMUL_PRECISION_FIELD
from .matmul_metal_runtime import SIDES
from .memory_access_report_io import MAX_REPORT_BYTES, read_json_object

GPU_TIMING_KIND = REPORT_KIND
CLOCK_FOR_TIME = (
    ("median_gpu_ns", EVENT),
    ("median_host_ns", SYNCHRONIZED),
    ("median_dispatch_ns", DISPATCH),
)


def _producer(digest: str) -> str:
    return (
        f"ai-journey-gpu-timing report {digest}: median_gpu_ns from {EVENT}, "
        f"median_host_ns from {SYNCHRONIZED}, median_dispatch_ns from {DISPATCH}, "
        "one launch and one clock per sample"
    )


def _matches_sources(report: Mapping[str, object], sources: Day74Sources) -> None:
    timed = tuple(report["config"]["launch"]["group_size"])
    if timed != sources.group_size:
        raise MatmulBottleneckError(
            f"the gpu-timing report launched threadgroups of {timed}, but the "
            f"matmul-metal report launched {sources.group_size}"
        )
    precision = report["environment"][MATMUL_PRECISION_FIELD]
    expected = sources.matmul_report["environment"][MATMUL_PRECISION_FIELD]
    if precision != expected:
        raise MatmulBottleneckError(
            f"the gpu-timing report ran at float32 matmul precision {precision}, "
            f"but the matmul-metal report ran at {expected}"
        )


def _timing_input(report: Mapping[str, object], sources: Day74Sources) -> TimingInput:
    """Convert a verified gpu-timing report into the rule's timing input."""

    _matches_sources(report, sources)
    entries = []
    for point in report["benchmark"]["points"]:
        shape = matmul_shape(point["m"], point["n"], point["k"])
        for side in SIDES:
            clocks = point["sides"][side]["clocks"]
            medians = {
                name: clocks[clock]["timing"]["median_ns"]
                for name, clock in CLOCK_FOR_TIME
            }
            entries.append(
                TimingEntry(
                    shape=shape,
                    kernel=side,
                    gpu_ns=medians["median_gpu_ns"],
                    host_ns=medians["median_host_ns"],
                    dispatch_ns=medians["median_dispatch_ns"],
                )
            )
    environment = report["environment"]
    return TimingInput(
        measured=report["measured"],
        producer=_producer(report[REPORT_DIGEST_FIELD]),
        environment=tuple((name, environment[name]) for name in ENVIRONMENT_FIELDS),
        repeats=report["config"]["sweep"]["repeats"],
        load_average_before=tuple(report["host_load"]["before"]),
        load_average_after=tuple(report["host_load"]["after"]),
        entries=tuple(entries),
    )


def _checked(load, *args, **kwargs):
    try:
        return load(*args, **kwargs)
    except GpuTimingVerificationError as error:
        raise MatmulBottleneckVerificationError(
            f"gpu-timing report: {error}"
        ) from error
    except GpuTimingError as error:
        raise MatmulBottleneckError(f"gpu-timing report: {error}") from error


def timing_from_gpu_timing_report(report: object, sources: Day74Sources) -> TimingInput:
    """Verify one in-memory gpu-timing report and convert it for the rule."""

    if type(sources) is not Day74Sources:
        raise TypeError("sources must be Day74Sources")
    _checked(verify_report, report)
    return _timing_input(report, sources)


def load_timing_source(path: Path, sources: Day74Sources) -> TimingInput:
    """Read a timing file of either kind; a gpu-timing report must verify.

    A gpu-timing report goes through its own verified, canonical loader; any
    other file goes through the bounded loader of the declared schema.
    """

    if type(sources) is not Day74Sources:
        raise TypeError("sources must be Day74Sources")
    try:
        payload = read_json_object(path, max_bytes=MAX_REPORT_BYTES, label="timing")
    except ValueError as error:
        raise MatmulBottleneckError(str(error)) from error
    if payload.get("kind") != GPU_TIMING_KIND:
        return load_timing(path)
    report, _ = _checked(load_gpu_timing_report, path)
    return _timing_input(report, sources)
