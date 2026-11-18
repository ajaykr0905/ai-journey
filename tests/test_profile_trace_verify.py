import copy
import json
from hashlib import sha256

import pytest

from ai_journey.memory_access_report import DIGEST_FIELD, seal_report
from ai_journey.profile_trace_events import (
    ProfileTraceError,
    parse_trace,
    parse_trace_bytes,
)
from ai_journey.profile_trace_layout import TraceLayoutError
from ai_journey.profile_trace_verify import (
    ProfileTraceVerificationError,
    sweep_gaps,
    sweep_op_times,
    verify_report,
)
from test_profile_trace_analysis import two_iterations
from test_profile_trace_capture import small_config
from test_profile_trace_events import PID, TID, document, span
from test_profile_trace_report import synthetic_report


def stored(tmp_path):
    """A synthetic report with its traces written beside it."""

    config, report, captured = synthetic_report()
    for item in captured:
        (tmp_path / item.file_name).write_bytes(item.trace_bytes)
    return config, report, captured


def reseal(report):
    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    return seal_report(unsigned)


def test_a_synthetic_report_verifies_against_its_traces(tmp_path):
    config, report, _ = stored(tmp_path)
    result = verify_report(report, trace_dir=tmp_path, config=config)

    assert result == {
        "report_sha256": report[DIGEST_FIELD],
        "measured": False,
        "scope": "synthetic_session",
        "shapes": 2,
        "bottlenecks": [
            "64x48x32:matmul_x_rows/synchronize",
            "33x17x9:matmul_x_rows/synchronize",
        ],
    }


def test_the_sweep_agrees_with_the_stack_on_a_nested_trace():
    trace = two_iterations()
    assert sweep_op_times(trace) == {
        "iteration": (2, 150_000, 30_000, 100_000),
        "side/dispatch": (2, 30_000, 22_000, 20_000),
        "aten::mm": (1, 8_000, 8_000, 8_000),
        "side/synchronize": (2, 90_000, 90_000, 60_000),
    }
    assert sweep_gaps(trace) == {(PID, TID): (30_000, 30_000)}
    equal = parse_trace(document(span("a", 0, 4), span("b", 0, 4), span("c", 5, 1)))
    assert sweep_op_times(equal)["b"][2] == 4_000
    assert sweep_op_times(equal)["a"][2] == 0
    assert sweep_gaps(equal) == {(PID, TID): (1_000, 1_000)}


def test_a_changed_trace_file_is_refused(tmp_path):
    _, report, captured = stored(tmp_path)
    path = tmp_path / captured[0].file_name
    path.write_bytes(captured[0].trace_bytes + b" ")
    with pytest.raises(ProfileTraceVerificationError, match="trace size"):
        verify_report(report, trace_dir=tmp_path)
    path.write_bytes(captured[0].trace_bytes.replace(b"Op count", b"Op cuont"))
    with pytest.raises(ProfileTraceVerificationError, match="trace digest"):
        verify_report(report, trace_dir=tmp_path)
    path.unlink()
    with pytest.raises(ProfileTraceError, match="readable regular"):
        verify_report(report, trace_dir=tmp_path)


def test_a_rebound_trace_still_needs_the_recorded_analysis(tmp_path):
    _, report, captured = stored(tmp_path)
    document = json.loads(captured[0].trace_bytes)
    for event in document["traceEvents"]:
        if event["name"] == "matmul_x_rows/synchronize":
            event["dur"] = 1.0
            break
    payload = json.dumps(document).encode()
    (tmp_path / captured[0].file_name).write_bytes(payload)
    forged = copy.deepcopy(report)
    forged["shapes"][0]["trace"]["sha256"] = sha256(payload).hexdigest()
    forged["shapes"][0]["trace"]["bytes"] = len(payload)
    with pytest.raises(ProfileTraceError):
        verify_report(reseal(forged), trace_dir=tmp_path)


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("measured",), True, "scope, label and boundary"),
        (("label",), "measurement", "scope, label and boundary"),
        (("profiler", "records_gpu_kernels"), True, "profiler"),
        (("environment", "torch_version"), "", "short strings"),
        (("load_average", "before"), [1.0, 2.0], "3 values"),
        (("load_average", "after"), [1.0, 2.0, -1.0], "finite number"),
        (("config_sha256",), "0" * 64, "config_sha256"),
        (("shapes", 0, "flops"), 1, "identity"),
        (("shapes", 0, "trace", "file"), "trace-0-1x1x1.json", "trace file must be"),
        (("shapes", 0, "trace", "events"), 1, "event count"),
        (("shapes", 0, "device_event_ns", "torch_matmul"), [1, 2], "device times"),
        (("shapes", 0, "device_event_ns", "torch_matmul"), [1, 2, -3], "device times"),
        (("shapes", 0, "check", "naive_same_bits"), False, "naive_same_bits"),
        (("shapes", 0, "check", "torch_disagreement"), 1.0, "within its bound"),
        (("shapes", 0, "check", "agreement_bound"), 1.0, "agreement bound"),
        (("shapes", 0, "analysis", "ops", 0, "self_ns"), 1, "analysis"),
        (("shapes", 1, "timeline"), "timeline", "timeline"),
        (("shapes", 1, "bottleneck", "top_op"), "aten::mm", "bottleneck"),
        (
            ("shapes", 1, "bottleneck", "sides", 0, "median_device_event_ns"),
            1,
            "bottleneck",
        ),
        (("summary", "bottlenecks", 0, "kind"), "host_other", "summary"),
        (("schema_version",), 2, "schema_version"),
        (("kind",), "other", "kind"),
    ],
)
def test_resealed_edits_are_refused(tmp_path, path, value, message):
    _, report, _ = stored(tmp_path)
    forged = copy.deepcopy(report)
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ProfileTraceVerificationError, match=message):
        verify_report(reseal(forged), trace_dir=tmp_path)


def test_digest_fields_and_config_are_checked(tmp_path):
    config, report, _ = stored(tmp_path)
    tampered = {**report, "label": "x"}
    with pytest.raises(ProfileTraceVerificationError, match="digest"):
        verify_report(tampered, trace_dir=tmp_path)
    with pytest.raises(ProfileTraceVerificationError, match="fields"):
        verify_report(reseal({**report, "extra": 1}), trace_dir=tmp_path)
    other = small_config(seed=76)
    with pytest.raises(ProfileTraceVerificationError, match="given config"):
        verify_report(report, trace_dir=tmp_path, config=other)
    shorter = copy.deepcopy(report)
    shorter["shapes"] = shorter["shapes"][:1]
    with pytest.raises(ProfileTraceVerificationError, match="one entry per"):
        verify_report(reseal(shorter), trace_dir=tmp_path)
    with pytest.raises(TypeError):
        verify_report(report, trace_dir=str(tmp_path))
    assert config.name == "profile-trace-test"


def test_a_trace_without_the_layout_is_refused_even_when_rebound(tmp_path):
    _, report, captured = stored(tmp_path)
    payload = captured[1].trace_bytes
    trace = parse_trace_bytes(payload)
    assert trace.event_count == report["shapes"][1]["trace"]["events"]
    document = json.loads(payload)
    document["traceEvents"] = [
        event
        for event in document["traceEvents"]
        if event["name"] != "torch_matmul/synchronize"
    ]
    changed = json.dumps(document).encode()
    (tmp_path / captured[1].file_name).write_bytes(changed)
    forged = copy.deepcopy(report)
    forged["shapes"][1]["trace"].update(
        sha256=sha256(changed).hexdigest(), bytes=len(changed)
    )
    with pytest.raises(TraceLayoutError, match="torch_matmul"):
        verify_report(reseal(forged), trace_dir=tmp_path)
