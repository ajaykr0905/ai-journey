import pytest

from ai_journey.matmul_metal_runtime import SIDES
from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.profile_trace_analysis import analyze_trace
from ai_journey.profile_trace_bottleneck import (
    BASIS,
    GPU_WAIT,
    HOST_DISPATCH,
    HOST_OTHER,
    bottleneck,
    bottleneck_kind,
    compulsory_bytes,
    no_reuse_bytes,
    side_figures,
)
from ai_journey.profile_trace_capture import capture_shape
from ai_journey.profile_trace_events import ProfileTraceError, parse_trace_bytes
from ai_journey.profile_trace_layout import ITERATION
from test_profile_trace_capture import DISPATCH_NS, WAIT_NS, FakeSession, small_config


def captured(index=0):
    config = small_config()
    item = capture_shape(FakeSession(), config, index)
    trace = parse_trace_bytes(item.trace_bytes)
    analysis = analyze_trace(trace, iteration_name=ITERATION, top_n=4, gap_count=2)
    return config, item, trace, analysis


def test_traffic_counts_follow_the_shape():
    shape = MatmulShape(2, 3, 4)
    assert compulsory_bytes(shape) == 4 * (8 + 12 + 6)
    assert no_reuse_bytes(shape) == 4 * (2 * 24 + 6)


@pytest.mark.parametrize(
    "name, kind",
    [
        ("matmul_x_rows/synchronize", GPU_WAIT),
        ("torch_matmul/synchronize", GPU_WAIT),
        ("matmul_x_columns/dispatch", HOST_DISPATCH),
        ("aten::mm", HOST_DISPATCH),
        ("profile_trace/iteration", HOST_OTHER),
        ("profile_trace/torch_matmul", HOST_OTHER),
        ("cuda/synchronize", HOST_OTHER),
    ],
)
def test_layout_names_map_to_kinds(name, kind):
    assert bottleneck_kind(name) == kind


def test_the_longest_wait_is_the_bottleneck_of_the_stand_in():
    config, item, trace, analysis = captured()
    result = bottleneck(analysis, trace, item.shape, config.ceilings, item.device_ns)

    assert result["basis"] == BASIS
    assert "no hardware counter" in BASIS
    assert result["top_op"] == "matmul_x_rows/synchronize"
    assert result["kind"] == GPU_WAIT
    assert result["self_ns"] == 3 * (WAIT_NS["matmul_x_rows"] + 2_000)
    hosts = sum(iteration.host_ns for iteration in analysis.iterations)
    assert result["share_of_iteration_host_time"] == result["self_ns"] / hosts
    assert [side["side"] for side in result["sides"]] == list(SIDES)


def test_side_figures_use_medians_and_the_day_74_ceilings():
    config, item, trace, _ = captured()
    figures = {
        row.side: row for row in side_figures(trace, item.shape, config.ceilings)
    }
    rows = figures["matmul_x_columns"]
    expected_host = (
        6_000 + DISPATCH_NS["matmul_x_columns"] + WAIT_NS["matmul_x_columns"]
    )

    assert rows.host_ns == expected_host
    assert rows.dispatch_ns == 2_000 + DISPATCH_NS["matmul_x_columns"]
    assert rows.synchronize_ns == 2_000 + WAIT_NS["matmul_x_columns"]
    assert rows.device_event_ns is None
    mapping = rows.to_mapping()
    flops = item.shape.flops
    assert mapping["host_observed_gflops_per_second"] == flops / expected_host
    assert mapping["fraction_of_fma_ceiling"] == flops / expected_host / 4618.0
    assert mapping["fraction_of_torch_matmul_ceiling"] == flops / expected_host / 3963.0
    assert mapping["fma_floor_ns"] == flops / 4618.0
    assert (
        mapping["compulsory_traffic_floor_ns"] == compulsory_bytes(item.shape) / 100.7
    )
    assert mapping["no_reuse_traffic_floor_ns"] == no_reuse_bytes(item.shape) / 100.7
    assert mapping["host_to_floor_ratio"]["fma_floor_ns"] == expected_host / (
        flops / 4618.0
    )
    assert mapping["synchronize_share_of_host"] == rows.synchronize_ns / expected_host
    with_events = side_figures(trace, item.shape, config.ceilings, item.device_ns)
    assert with_events[0].device_event_ns == DISPATCH_NS["matmul_x_columns"] + 2_000


def test_a_bottleneck_needs_top_ops_and_layout_spans():
    config, item, trace, analysis = captured()
    with pytest.raises(ProfileTraceError, match="needs an analysis"):
        bottleneck({}, trace, item.shape, config.ceilings)
    bare = parse_trace_bytes(
        b'{"traceEvents": [{"ph": "X", "name": "profile_trace/iteration", '
        b'"pid": 1, "tid": 1, "ts": 0, "dur": 1}]}'
    )
    with pytest.raises(ProfileTraceError, match="no 'profile_trace/matmul_x_columns'"):
        side_figures(bare, item.shape, config.ceilings)
