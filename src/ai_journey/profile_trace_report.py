"""Seal one profiler capture into a report that binds its trace files by SHA-256.

The report embeds the config and its digest, the host environment, the load
averages before and after the capture, and one entry per shape. Each entry
names its stored trace file with the file's SHA-256 digest and byte count,
so a verifier can find the exact bytes that the analysis came from. The
entry also holds the analysis, the text timeline, the output check, the MPS
event times and the bottleneck inference.

Only ``TorchProfilerSession`` makes a measured report, labeled as a
host-specific Apple GPU (Metal) measurement. Any other session gives
``measured: false`` and a boundary that denies a measurement. The seal is
the shared SHA-256 digest over canonical JSON; it detects drift and
tampering, but it is not a signature.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from hashlib import sha256

from .matmul_metal_report import environment
from .matmul_metal_runtime import SIDES
from .memory_access_report import seal_report
from .profile_trace_analysis import analyze_trace
from .profile_trace_bottleneck import bottleneck
from .profile_trace_capture import (
    ACTIVITY,
    DEVICE_TIMER,
    CapturedShape,
    TorchProfilerSession,
    capture_traces,
)
from .profile_trace_config import ProfileTraceConfig
from .profile_trace_events import ProfileTraceError, parse_trace_bytes
from .profile_trace_layout import ITERATION, check_layout
from .profile_trace_timeline import render_timeline

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "profile_trace_report"
MEASURED_SCOPE = "apple_gpu_metal_host_trace_this_host"
SYNTHETIC_SCOPE = "synthetic_session"
MEASURED_LABEL = "host-specific Apple GPU (Metal) measurement"
SYNTHETIC_LABEL = "synthetic stand-in session; not a measurement"
PROFILER = {
    "activity": ACTIVITY,
    "device_timer": DEVICE_TIMER,
    "records_gpu_kernels": False,
    "statement": (
        "ProfilerActivity.CPU records only host-side spans: the dispatch calls "
        "and the host's wait inside torch.mps.synchronize(). It records no GPU "
        "kernel and no GPU kernel time. The GPU work of a side shows only as "
        "host wait time in its synchronize span."
    ),
}
MEASURED_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "hardware_counters_read": False,
    "portable_performance_claim": False,
    "statement": (
        "Host-side torch.profiler spans and MPS event times taken on the Apple "
        "GPU (Metal) of the host that built this report, under its load. The "
        "bottleneck is an inference from these timings and the measured Day 74 "
        "ceilings, not a reading of hardware counters. Nothing here measures "
        "CUDA, NVIDIA, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_session": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "hardware_counters_read": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in session wrote these traces on its own clock, so every time "
        "in this report is synthetic and not a measurement of any host."
    ),
}


def load_average(reader: Callable[[], tuple[float, ...]] = os.getloadavg) -> list:
    """Return the 1, 5 and 15 minute load averages, rounded to two decimals."""

    values = tuple(reader())
    if len(values) != 3 or not all(
        type(value) is float and math.isfinite(value) and value >= 0 for value in values
    ):
        raise ProfileTraceError("load averages must be three finite floats >= 0")
    return [round(value, 2) for value in values]


def shape_entry(config: ProfileTraceConfig, item: CapturedShape) -> dict[str, object]:
    """Analyze one captured shape and return its report entry.

    The analysis reads the stored bytes, the same bytes the digest binds,
    so the verifier can recompute it from the file alone.
    """

    if not isinstance(item, CapturedShape):
        raise TypeError("a shape entry needs a CapturedShape")
    controls = config.analysis
    trace = parse_trace_bytes(item.trace_bytes)
    check_layout(trace, config.iterations)
    analysis = analyze_trace(
        trace,
        iteration_name=ITERATION,
        top_n=controls.top_n,
        gap_count=controls.gap_count,
    )
    shape = item.shape
    return {
        "index": item.index,
        **shape.to_mapping(),
        "flops": shape.flops,
        "trace": {
            "file": item.file_name,
            "sha256": sha256(item.trace_bytes).hexdigest(),
            "bytes": len(item.trace_bytes),
            "events": trace.event_count,
        },
        "device_event_ns": {side: list(item.device_ns[side]) for side in SIDES},
        "check": {
            "naive_same_bits": item.naive_same_bits,
            "torch_disagreement": item.torch_disagreement,
            "agreement_bound": item.agreement_bound,
        },
        "analysis": analysis.to_mapping(),
        "timeline": render_timeline(
            trace, width=controls.timeline_width, depth_limit=controls.timeline_depth
        ),
        "bottleneck": bottleneck(
            analysis, trace, shape, config.ceilings, item.device_ns
        ),
    }


def summary(shapes: list[dict[str, object]]) -> dict[str, object]:
    """Name the bottleneck of every shape, so a reader sees it in one place."""

    return {
        "bottlenecks": [
            {
                "m": entry["m"],
                "n": entry["n"],
                "k": entry["k"],
                "top_op": entry["bottleneck"]["top_op"],
                "kind": entry["bottleneck"]["kind"],
                "share_of_iteration_host_time": entry["bottleneck"][
                    "share_of_iteration_host_time"
                ],
            }
            for entry in shapes
        ]
    }


def build_report(
    config: ProfileTraceConfig,
    captured: tuple[CapturedShape, ...],
    *,
    session: object,
    load_before: list,
    load_after: list,
) -> dict[str, object]:
    """Assemble and seal the report of one finished capture."""

    if not isinstance(config, ProfileTraceConfig):
        raise TypeError("config must be a ProfileTraceConfig")
    if len(captured) != len(config.shapes) or any(
        item.index != index or item.shape != config.shapes[index]
        for index, item in enumerate(captured)
    ):
        raise ProfileTraceError("captured shapes must follow the config in order")
    measured = type(session) is TorchProfilerSession and session.measured is True
    shapes = [shape_entry(config, item) for item in captured]
    return seal_report(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "measured": measured,
            "scope": MEASURED_SCOPE if measured else SYNTHETIC_SCOPE,
            "label": MEASURED_LABEL if measured else SYNTHETIC_LABEL,
            "boundary": MEASURED_BOUNDARY if measured else SYNTHETIC_BOUNDARY,
            "profiler": PROFILER,
            "config": config.to_mapping(),
            "config_sha256": config.fingerprint(),
            "environment": environment(session),
            "load_average": {"before": list(load_before), "after": list(load_after)},
            "shapes": shapes,
            "summary": summary(shapes),
        }
    )


def measure_report(
    config: ProfileTraceConfig,
    *,
    session: object | None = None,
    load_reader: Callable[[], tuple[float, ...]] = os.getloadavg,
) -> tuple[dict[str, object], tuple[CapturedShape, ...]]:
    """Capture every shape and return the sealed report with the captures.

    Without a session this starts ``TorchProfilerSession``, which raises the
    Day 74 unavailable error on a host without MPS before any work runs.
    """

    active = TorchProfilerSession() if session is None else session
    before = load_average(load_reader)
    captured = capture_traces(config, active)
    after = load_average(load_reader)
    report = build_report(
        config, captured, session=active, load_before=before, load_after=after
    )
    return report, captured
