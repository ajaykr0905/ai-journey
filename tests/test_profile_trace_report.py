from hashlib import sha256

import pytest

from ai_journey.memory_access_plan import canonical_json_bytes
from ai_journey.memory_access_report import DIGEST_FIELD, report_digest
from ai_journey.profile_trace_capture import capture_traces
from ai_journey.profile_trace_events import ProfileTraceError
from ai_journey.profile_trace_report import (
    MEASURED_SCOPE,
    PROFILER,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
    build_report,
    load_average,
    measure_report,
)
from test_profile_trace_capture import FakeSession, small_config


def synthetic_report():
    config = small_config()
    loads = iter([(1.234, 2.0, 3.0), (4.0, 5.0, 6.555)])
    report, captured = measure_report(
        config, session=FakeSession(), load_reader=lambda: next(loads)
    )
    return config, report, captured


def test_a_stand_in_session_makes_a_sealed_synthetic_report():
    config, report, captured = synthetic_report()

    assert report[DIGEST_FIELD] == report_digest(report)
    assert report["measured"] is False
    assert report["scope"] == SYNTHETIC_SCOPE != MEASURED_SCOPE
    assert report["label"] == SYNTHETIC_LABEL
    assert report["boundary"] == SYNTHETIC_BOUNDARY
    assert report["profiler"] == PROFILER
    assert report["profiler"]["records_gpu_kernels"] is False
    assert report["config"] == config.to_mapping()
    assert report["config_sha256"] == config.fingerprint()
    assert report["environment"]["torch_version"] == "2.15.0"
    assert report["load_average"] == {
        "before": [1.23, 2.0, 3.0],
        "after": [4.0, 5.0, 6.55],
    }
    canonical_json_bytes(report)


def test_each_shape_binds_its_trace_bytes_and_carries_the_analysis():
    _, report, captured = synthetic_report()

    for entry, item in zip(report["shapes"], captured):
        assert entry["trace"] == {
            "file": item.file_name,
            "sha256": sha256(item.trace_bytes).hexdigest(),
            "bytes": len(item.trace_bytes),
            "events": entry["trace"]["events"],
        }
        assert entry["flops"] == item.shape.flops
        assert entry["device_event_ns"]["matmul_x_rows"] == list(
            item.device_ns["matmul_x_rows"]
        )
        assert entry["check"]["naive_same_bits"] is True
        assert entry["analysis"]["iteration_host_ns"]["count"] == 3
        assert entry["timeline"].startswith("timeline extent_ns=")
        assert entry["bottleneck"]["top_op"] == "matmul_x_rows/synchronize"
    assert report["summary"]["bottlenecks"][1] == {
        "m": 33,
        "n": 17,
        "k": 9,
        "top_op": "matmul_x_rows/synchronize",
        "kind": "gpu_wait",
        "share_of_iteration_host_time": report["shapes"][1]["bottleneck"][
            "share_of_iteration_host_time"
        ],
    }


def test_captures_must_follow_the_config():
    config = small_config()
    captured = capture_traces(config, FakeSession())
    loads = {"load_before": [0.0] * 3, "load_after": [0.0] * 3}
    with pytest.raises(ProfileTraceError, match="follow the config"):
        build_report(config, captured[::-1], session=FakeSession(), **loads)
    with pytest.raises(ProfileTraceError, match="follow the config"):
        build_report(config, captured[:1], session=FakeSession(), **loads)
    with pytest.raises(TypeError):
        build_report({}, captured, session=FakeSession(), **loads)


def test_a_session_that_claims_to_measure_is_still_synthetic():
    config = small_config()
    session = FakeSession()
    session.measured = True
    captured = capture_traces(config, session)
    report = build_report(
        config, captured, session=session, load_before=[0.0] * 3, load_after=[0.0] * 3
    )
    assert report["measured"] is False


@pytest.mark.parametrize(
    "values", [(1.0, 2.0), (1.0, 2.0, float("nan")), (1, 2, 3), (-1.0, 0.0, 0.0)]
)
def test_load_averages_are_checked(values):
    with pytest.raises(ProfileTraceError, match="load averages"):
        load_average(lambda: values)
    assert len(load_average()) == 3
