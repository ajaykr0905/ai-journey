"""Verify a sealed profiler report against the trace files it binds.

The verifier never imports torch and never calls the capture. For each
shape, the steps are:

1. Read the named trace file from the trace directory with the bounded
   reader, and require its byte count and SHA-256 digest.
2. Parse it with the strict parser, require the capture layout, and require
   the redacted host name and export path that the capture writes.
3. Recompute the analysis, the timeline and the bottleneck from the bytes,
   and require each section to match the report as canonical JSON.
4. Cross-check the op times and the gaps with a second method: a sweep over
   the span boundaries of each thread gives every elementary interval to
   its innermost span. This method shares no code with the stack-based
   self time of ``profile_trace_analysis``, so each checks the other.

The verifier cannot observe a timing again. It checks the MPS event times,
the load averages and the output check for structure and bounds only. The
config parser is shared, and it only checks bounds.
"""

from __future__ import annotations

import heapq
import math
from hashlib import sha256
from pathlib import Path

from .matmul_metal_benchmark import agreement_bound
from .matmul_metal_runtime import SIDES
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD
from .profile_trace_analysis import WINDOW_CATEGORY, analyze_trace
from .profile_trace_bottleneck import bottleneck
from .profile_trace_capture import REDACTED_HOST
from .profile_trace_config import ProfileTraceConfig, config_from_mapping
from .profile_trace_events import (
    ParsedTrace,
    ProfileTraceError,
    TraceSpan,
    decode_trace_json,
    parse_trace_bytes,
    read_trace_bytes,
)
from .profile_trace_layout import ITERATION, check_layout, trace_file_name
from .profile_trace_report import (
    MEASURED_BOUNDARY,
    MEASURED_LABEL,
    MEASURED_SCOPE,
    PROFILER,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
    summary,
)
from .profile_trace_timeline import render_timeline

_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "scope",
    "label",
    "boundary",
    "profiler",
    "config",
    "config_sha256",
    "environment",
    "load_average",
    "shapes",
    "summary",
    DIGEST_FIELD,
}
_SHAPE_FIELDS = {
    "index",
    "m",
    "n",
    "k",
    "flops",
    "trace",
    "device_event_ns",
    "check",
    "analysis",
    "timeline",
    "bottleneck",
}
MAX_TOKEN = 64


class ProfileTraceVerificationError(ProfileTraceError):
    """Raised when a report was read but does not match what it claims."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileTraceVerificationError(message)


def _same(actual: object, expected: object, label: str) -> None:
    """Compare as canonical JSON, so ``1``, ``1.0`` and ``true`` stay different."""

    _require(
        canonical_json_bytes(actual) == canonical_json_bytes(expected),
        f"{label} does not match its recomputation",
    )


def _digest(report: dict[str, object]) -> str:
    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    return sha256(canonical_json_bytes(unsigned)).hexdigest()


def sweep_op_times(trace: ParsedTrace) -> dict[str, tuple[int, int, int, int]]:
    """Return ``name -> (calls, total, self, max)`` by an interval sweep.

    On each thread, every interval between two consecutive span boundaries
    belongs to the innermost span open over it: the latest start, then the
    shortest span, then the later one in the file.
    """

    rows: dict[str, list[int]] = {}
    threads: dict[object, list[tuple[int, TraceSpan]]] = {}
    for position, span in enumerate(trace.spans):
        if span.category == WINDOW_CATEGORY:
            continue
        threads.setdefault(span.thread, []).append((position, span))
        row = rows.setdefault(span.name, [0, 0, 0, 0])
        row[0] += 1
        row[1] += span.duration_ns
        row[3] = max(row[3], span.duration_ns)
    for members in threads.values():
        points = sorted(
            {span.start_ns for _, span in members}
            | {span.end_ns for _, span in members}
        )
        # A max-heap of the started spans; one that has ended leaves only
        # when it reaches the top, so each span enters and leaves once.
        starting = sorted(members, key=lambda item: item[1].start_ns)
        heap: list[tuple[int, int, int, int, str]] = []
        cursor = 0
        for low, high in zip(points, points[1:]):
            while cursor < len(starting) and starting[cursor][1].start_ns <= low:
                position, span = starting[cursor]
                key = (-span.start_ns, span.duration_ns, -position)
                heapq.heappush(heap, (*key, span.end_ns, span.name))
                cursor += 1
            while heap and heap[0][3] < high:
                heapq.heappop(heap)
            if heap:
                rows[heap[0][4]][2] += high - low
    return {name: tuple(values) for name, values in rows.items()}


def sweep_gaps(trace: ParsedTrace) -> dict[object, tuple[int, int]]:
    """Return ``thread -> (total gap, longest gap)`` from the union of its spans."""

    threads: dict[object, list[TraceSpan]] = {}
    for span in trace.spans:
        if span.category != WINDOW_CATEGORY:
            threads.setdefault(span.thread, []).append(span)
    result = {}
    for thread, spans in threads.items():
        spans.sort(key=lambda span: (span.start_ns, span.end_ns))
        covered_end = spans[0].end_ns
        gaps = []
        for span in spans[1:]:
            if span.start_ns > covered_end:
                gaps.append(span.start_ns - covered_end)
            covered_end = max(covered_end, span.end_ns)
        result[thread] = (sum(gaps), max(gaps, default=0))
    return result


def _cross_check(trace: ParsedTrace, analysis: dict[str, object], label: str) -> None:
    swept = sweep_op_times(trace)
    stacked = {
        row["name"]: (row["calls"], row["total_ns"], row["self_ns"], row["max_ns"])
        for row in analysis["ops"]
    }
    _require(swept == stacked, f"{label} op times disagree with the interval sweep")
    gaps = sweep_gaps(trace)
    for thread in analysis["threads"]:
        key = (thread["pid"], thread["tid"])
        _require(
            gaps.get(key) == (thread["total_gap_ns"], thread["max_gap_ns"]),
            f"{label} gaps disagree with the span union",
        )
    _require(len(gaps) == len(analysis["threads"]), f"{label} thread count differs")


def _number(value: object, label: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{label} must be a finite number >= 0",
    )
    return float(value)


def _check_header(report: dict[str, object]) -> bool:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    _require(report[DIGEST_FIELD] == _digest(report), "report digest does not match")
    _require(
        report["schema_version"] == REPORT_SCHEMA_VERSION
        and type(report["schema_version"]) is int,
        f"schema_version must be {REPORT_SCHEMA_VERSION}",
    )
    _require(report["kind"] == REPORT_KIND, f"kind must be {REPORT_KIND}")
    measured = report["measured"]
    _require(type(measured) is bool, "measured must be a boolean")
    expected = (
        (MEASURED_SCOPE, MEASURED_LABEL, MEASURED_BOUNDARY)
        if measured
        else (SYNTHETIC_SCOPE, SYNTHETIC_LABEL, SYNTHETIC_BOUNDARY)
    )
    _same(
        [report["scope"], report["label"], report["boundary"]],
        list(expected),
        "scope, label and boundary",
    )
    _same(report["profiler"], PROFILER, "profiler")
    environment = report["environment"]
    _require(
        type(environment) is dict
        and all(
            type(key) is str and type(value) is str and 0 < len(value) <= MAX_TOKEN
            for key, value in environment.items()
        ),
        "environment must map names to short strings",
    )
    loads = report["load_average"]
    _require(
        type(loads) is dict and set(loads) == {"before", "after"},
        "load_average must hold before and after",
    )
    for key in ("before", "after"):
        values = loads[key]
        _require(
            type(values) is list and len(values) == 3, f"load {key} needs 3 values"
        )
        for value in values:
            _number(value, f"load {key}")
    return measured


def _check_redacted(document: dict[str, object], name: str, label: str) -> None:
    """Require the host fields that the capture replaces to hold its values.

    A digest only binds bytes, so without this check a stored trace that
    still names the host or a local path would verify.
    """

    expected = {"host_name": REDACTED_HOST, "traceName": name}
    for key, value in expected.items():
        _require(
            document.get(key, value) == value,
            f"{label} trace {key} must be redacted to {value!r}",
        )


def _check_shape(
    config: ProfileTraceConfig, index: int, entry: object, trace_dir: Path
) -> None:
    label = f"shape {index}"
    _require(
        type(entry) is dict and set(entry) == _SHAPE_FIELDS,
        f"{label} fields do not match the schema",
    )
    shape = config.shapes[index]
    _same(
        {key: entry[key] for key in ("index", "m", "n", "k", "flops")},
        {"index": index, **shape.to_mapping(), "flops": shape.flops},
        f"{label} identity",
    )
    stored = entry["trace"]
    name = trace_file_name(index, shape)
    _require(
        type(stored) is dict and set(stored) == {"file", "sha256", "bytes", "events"},
        f"{label} trace fields do not match the schema",
    )
    _require(stored["file"] == name, f"{label} trace file must be {name}")
    payload = read_trace_bytes(trace_dir / name)
    _require(
        stored["bytes"] == len(payload) and type(stored["bytes"]) is int,
        f"{label} trace size differs from the report",
    )
    _require(
        stored["sha256"] == sha256(payload).hexdigest(),
        f"{label} trace digest differs from the report",
    )
    try:
        trace = parse_trace_bytes(payload)
        check_layout(trace, config.iterations)
    except ProfileTraceError as error:
        raise ProfileTraceVerificationError(f"{label} trace: {error}") from error
    _check_redacted(decode_trace_json(payload), name, label)
    _same(stored["events"], trace.event_count, f"{label} event count")
    device = entry["device_event_ns"]
    _require(
        type(device) is dict and list(device) == list(SIDES),
        f"{label} device_event_ns must hold every side in order",
    )
    for side, values in device.items():
        _require(
            type(values) is list
            and len(values) == config.iterations
            and all(type(value) is int and value >= 0 for value in values),
            f"{label} {side} device times must be {config.iterations} integers >= 0",
        )
    check = entry["check"]
    bound = agreement_bound(shape.k)
    _require(
        type(check) is dict
        and set(check) == {"naive_same_bits", "torch_disagreement", "agreement_bound"},
        f"{label} check fields do not match the schema",
    )
    _same(check["naive_same_bits"], True, f"{label} naive_same_bits")
    _same(check["agreement_bound"], bound, f"{label} agreement bound")
    disagreement = check["torch_disagreement"]
    _require(
        type(disagreement) is float and 0 <= disagreement <= bound,
        f"{label} torch disagreement must be a float within its bound",
    )
    controls = config.analysis
    analysis = analyze_trace(
        trace,
        iteration_name=ITERATION,
        top_n=controls.top_n,
        gap_count=controls.gap_count,
    )
    mapping = analysis.to_mapping()
    _cross_check(trace, mapping, label)
    _same(entry["analysis"], mapping, f"{label} analysis")
    _same(
        entry["timeline"],
        render_timeline(
            trace, width=controls.timeline_width, depth_limit=controls.timeline_depth
        ),
        f"{label} timeline",
    )
    events = {side: tuple(values) for side, values in device.items()}
    _same(
        entry["bottleneck"],
        bottleneck(analysis, trace, shape, config.ceilings, events),
        f"{label} bottleneck",
    )


def verify_report(
    report: dict[str, object],
    *,
    trace_dir: Path,
    config: ProfileTraceConfig | None = None,
) -> dict[str, object]:
    """Verify every section of one report; return a short summary of it.

    With ``config``, the report must also come from exactly that config.
    """

    if not isinstance(trace_dir, Path):
        raise TypeError("trace_dir must be pathlib.Path")
    measured = _check_header(report)
    embedded = config_from_mapping(report["config"])
    _same(report["config"], embedded.to_mapping(), "config")
    _same(report["config_sha256"], embedded.fingerprint(), "config_sha256")
    if config is not None:
        _require(
            isinstance(config, ProfileTraceConfig)
            and config.fingerprint() == embedded.fingerprint(),
            "report does not come from the given config",
        )
    shapes = report["shapes"]
    _require(
        type(shapes) is list and len(shapes) == len(embedded.shapes),
        "report must hold one entry per configured shape",
    )
    for index, entry in enumerate(shapes):
        _check_shape(embedded, index, entry, trace_dir)
    _same(report["summary"], summary(shapes), "summary")
    return {
        "report_sha256": report[DIGEST_FIELD],
        "measured": measured,
        "scope": report["scope"],
        "shapes": len(shapes),
        "bottlenecks": [
            f"{item['m']}x{item['n']}x{item['k']}:{item['top_op']}"
            for item in report["summary"]["bottlenecks"]
        ],
    }
