import pytest

from ai_journey.memory_access_tiling import MatmulShape
from ai_journey.profile_trace_events import parse_trace
from ai_journey.profile_trace_layout import (
    ITERATION,
    TraceLayoutError,
    check_layout,
    phase_region,
    require_trace_file_name,
    side_order,
    side_region,
    trace_file_name,
)
from test_profile_trace_events import document, span


def layout_events(iterations, *, tid=1, start=0):
    """Spans in the capture layout, 10 us per phase, 1 us apart."""

    events = []
    now = start
    for number in range(iterations):
        begin = now
        now += 1
        for side in side_order(number):
            side_begin = now
            for phase in ("dispatch", "synchronize"):
                events.append(span(phase_region(side, phase), now, 10, tid=tid))
                now += 10
            events.append(
                span(side_region(side), side_begin, now - side_begin, tid=tid)
            )
        now += 1
        events.append(span(ITERATION, begin, now - begin, tid=tid))
        now += 5
    return events


def test_names_orders_and_file_names():
    assert side_region("torch_matmul") == "profile_trace/torch_matmul"
    assert phase_region("matmul_x_rows", "synchronize") == "matmul_x_rows/synchronize"
    assert side_order(0) == ("matmul_x_columns", "matmul_x_rows", "torch_matmul")
    assert side_order(1) == ("matmul_x_rows", "torch_matmul", "matmul_x_columns")
    assert side_order(3) == side_order(0)
    assert trace_file_name(2, MatmulShape(2048, 1, 33)) == "trace-2-2048x1x33.json"
    for bad in ("../trace-0-1x1x1.json", "trace-0-1x1x1.json/", "x.json", 1):
        with pytest.raises(TraceLayoutError, match="file names"):
            require_trace_file_name(bad)
    with pytest.raises(TraceLayoutError, match="side must be"):
        side_region("cuda")
    with pytest.raises(TraceLayoutError, match="phase must be"):
        phase_region("torch_matmul", "launch")


def test_the_capture_layout_passes_and_torch_dispatch_may_hold_aten_ops():
    events = layout_events(3)
    torch_dispatch = next(
        event for event in events if event["name"] == "torch_matmul/dispatch"
    )
    events.append(span("aten::mm", torch_dispatch["ts"] + 1, 2, tid=1, cat="cpu_op"))
    check_layout(parse_trace(document(*events)), 3)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda events: events[:-1], "holds 2"),
        (lambda events: events + layout_events(1, start=500), "holds 4"),
        (lambda events: events[:-1] + layout_events(1, tid=2, start=500), "one thread"),
        (
            lambda events: [e for e in events if "matmul_x_rows" not in e["name"]],
            "holds",
        ),
        (
            lambda events: [
                e for e in events if e["name"] != "torch_matmul/synchronize"
            ],
            "side torch_matmul holds",
        ),
        (lambda events: events + [span("aten::copy_", 2, 1, tid=1)], "must not hold"),
        (lambda events: events + [span("x", 12, 1, tid=1)], "must not hold"),
    ],
)
def test_traces_without_the_layout_are_refused(mutate, message):
    events = mutate(layout_events(3))
    with pytest.raises(TraceLayoutError, match=message):
        check_layout(parse_trace(document(*events)), 3)


def test_iterations_must_be_top_level():
    events = layout_events(2) + [span("outer", 0, 10_000, tid=1)]
    with pytest.raises(TraceLayoutError, match="top-level"):
        check_layout(parse_trace(document(*events)), 2)
    with pytest.raises(TypeError):
        check_layout({}, 2)
