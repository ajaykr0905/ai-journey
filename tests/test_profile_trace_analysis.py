import pytest

from ai_journey.profile_trace_analysis import (
    MAX_TOP_N,
    OpStats,
    op_stats,
    self_times,
    top_ops,
)
from ai_journey.profile_trace_events import ProfileTraceError, parse_trace
from test_profile_trace_events import TORCH_LAYOUT, document, span

from ai_journey.profile_trace_events import parse_trace_bytes


def two_iterations():
    """Two iterations of a dispatch and a wait, with one nested aten call."""

    return parse_trace(
        document(
            span("iteration", 0, 100),
            span("side/dispatch", 5, 20),
            span("aten::mm", 10, 8, cat="cpu_op"),
            span("side/synchronize", 30, 60),
            span("iteration", 130, 50),
            span("side/dispatch", 131, 10),
            span("side/synchronize", 145, 30),
            span("window", 0, 200, tid="PyTorch Profiler", cat="Trace"),
        )
    )


def test_self_time_subtracts_only_direct_children():
    trace = two_iterations()
    rows = {
        (item.span.name, item.span.start_ns): own
        for item, own in zip(trace.nested, self_times(trace))
    }
    assert rows[("iteration", 0)] == 20_000
    assert rows[("side/dispatch", 5000)] == 12_000
    assert rows[("aten::mm", 10_000)] == 8_000
    assert rows[("iteration", 130_000)] == 10_000
    assert rows[("window", 0)] == 200_000


def test_ops_add_spans_by_name_and_leave_out_the_recording_window():
    stats = {row.name: row for row in op_stats(two_iterations())}

    assert set(stats) == {"iteration", "side/dispatch", "aten::mm", "side/synchronize"}
    assert stats["iteration"] == OpStats("iteration", 2, 150_000, 30_000, 100_000)
    assert stats["side/dispatch"] == OpStats("side/dispatch", 2, 30_000, 22_000, 20_000)
    assert stats["side/synchronize"].self_ns == 90_000
    assert sum(row.self_ns for row in stats.values()) == 150_000
    assert stats["aten::mm"].to_mapping() == {
        "name": "aten::mm",
        "calls": 1,
        "total_ns": 8000,
        "self_ns": 8000,
        "max_ns": 8000,
    }


def test_recursive_ops_count_self_time_once():
    trace = parse_trace(document(span("f", 0, 10), span("f", 1, 4)))
    (row,) = op_stats(trace)
    assert (row.calls, row.total_ns, row.self_ns, row.max_ns) == (
        2,
        14_000,
        10_000,
        10_000,
    )


def test_top_ops_rank_by_self_time_then_name():
    stats = op_stats(two_iterations())
    ranked = top_ops(stats, 3)
    assert [row.name for row in ranked] == [
        "side/synchronize",
        "iteration",
        "side/dispatch",
    ]
    tie = (OpStats("b", 1, 5, 5, 5), OpStats("a", 1, 5, 5, 5))
    assert [row.name for row in top_ops(tie, 2)] == ["a", "b"]
    assert len(top_ops(stats, MAX_TOP_N)) == 4
    for count in (0, MAX_TOP_N + 1, True):
        with pytest.raises(ProfileTraceError, match="top_n"):
            top_ops(stats, count)
    with pytest.raises(TypeError):
        top_ops(list(stats), 1)


def test_the_torch_layout_has_annotations_and_aten_ops():
    stats = {row.name: row for row in op_stats(parse_trace_bytes(TORCH_LAYOUT))}
    assert set(stats) == {"iter_0", "aten::mm"}
    assert stats["iter_0"].self_ns == 7532139 - 214868


@pytest.mark.parametrize(
    "fields, message",
    [
        (("", 1, 1, 1, 1), "non-empty name"),
        (("a", 0, 1, 1, 1), "calls"),
        (("a", 1, 1, 2, 1), "inconsistent"),
        (("a", 1, 1, 1, 2), "inconsistent"),
        (("a", 1, -1, 0, 0), "total_ns"),
        (("a", True, 1, 1, 1), "calls"),
    ],
)
def test_op_stats_are_validated(fields, message):
    with pytest.raises(ProfileTraceError, match=message):
        OpStats(*fields)


def test_self_times_need_a_parsed_trace():
    with pytest.raises(TypeError):
        self_times({})
