import pytest

from ai_journey.profile_trace_analysis import (
    MAX_TOP_N,
    analyze_trace,
    iteration_times,
    median_low,
    OpStats,
    op_stats,
    origin_ns,
    self_times,
    top_level_gaps,
    top_ops,
)
from ai_journey.profile_trace_events import ProfileTraceError, parse_trace
from test_profile_trace_events import PID, TID, TORCH_LAYOUT, document, span

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


def test_gaps_lie_between_consecutive_top_level_calls_per_thread():
    trace = parse_trace(
        document(
            span("iteration", 10, 100),
            span("inner", 20, 10),
            span("iteration", 130, 50),
            span("iteration", 185, 5),
            span("other", 0, 3, tid=9),
            span("window", 0, 500, tid="PyTorch Profiler", cat="Trace"),
        )
    )
    assert origin_ns(trace) == 0
    main, other = top_level_gaps(trace)

    assert (main.pid, main.tid, main.calls) == (PID, TID, 3)
    assert main.busy_ns == 155_000
    assert main.extent_ns == 180_000
    assert main.total_gap_ns == main.extent_ns - main.busy_ns == 25_000
    assert [gap.gap_ns for gap in main.gaps] == [20_000, 5_000]
    first = main.gaps[0]
    assert (first.before, first.after, first.offset_ns) == (
        "iteration",
        "iteration",
        110_000,
    )
    assert main.largest(1) == (first,)
    summary = main.to_mapping(4)
    assert summary["gaps"] == 2 and summary["max_gap_ns"] == 20_000
    assert summary["largest_gaps"][1] == {
        "before": "iteration",
        "after": "iteration",
        "offset_ns": 180_000,
        "gap_ns": 5_000,
    }
    assert other.to_mapping(1)["max_gap_ns"] == 0
    assert other.gaps == ()


def test_equal_gaps_list_the_earliest_first_and_counts_are_bounded():
    trace = parse_trace(
        document(span("a", 0, 1), span("b", 2, 1), span("c", 4, 1)),
    )
    (thread,) = top_level_gaps(trace)
    assert [gap.after for gap in thread.largest(2)] == ["b", "c"]
    with pytest.raises(ProfileTraceError, match="gap count"):
        thread.largest(0)


def test_a_trace_without_ops_has_no_origin():
    trace = parse_trace(document(span("window", 0, 5, cat="Trace")))
    with pytest.raises(ProfileTraceError, match="no op spans"):
        top_level_gaps(trace)


def test_iterations_report_host_time_and_the_split_of_their_children():
    first, second = iteration_times(two_iterations(), "iteration")

    assert first.to_mapping() == {
        "index": 0,
        "offset_ns": 0,
        "host_ns": 100_000,
        "children_ns": {"side/dispatch": 20_000, "side/synchronize": 60_000},
        "uncovered_ns": 20_000,
    }
    assert (second.index, second.offset_ns, second.host_ns) == (1, 130_000, 50_000)
    assert second.uncovered_ns == 10_000
    assert iteration_times(two_iterations(), "absent") == ()
    with pytest.raises(ProfileTraceError, match="non-empty"):
        iteration_times(two_iterations(), "")


def test_median_low_stays_an_exact_integer():
    assert median_low((5, 1, 3)) == 3
    assert median_low((4, 1, 3, 2)) == 2
    with pytest.raises(ProfileTraceError, match="at least one"):
        median_low(())


def test_the_whole_analysis_maps_to_report_data():
    analysis = analyze_trace(
        two_iterations(), iteration_name="iteration", top_n=2, gap_count=1
    )
    mapping = analysis.to_mapping()

    assert analysis.origin_ns == 0
    assert [row["name"] for row in mapping["top_ops_by_self_time"]] == [
        "side/synchronize",
        "iteration",
    ]
    assert len(mapping["ops"]) == 4
    (thread,) = mapping["threads"]
    assert thread["largest_gaps"] == [
        {
            "before": "iteration",
            "after": "iteration",
            "offset_ns": 100_000,
            "gap_ns": 30_000,
        }
    ]
    assert mapping["iteration_name"] == "iteration"
    assert mapping["iteration_host_ns"] == {
        "count": 2,
        "min": 50_000,
        "median_low": 50_000,
        "max": 100_000,
        "total": 150_000,
    }
    with pytest.raises(ProfileTraceError, match="no 'absent' spans"):
        analyze_trace(two_iterations(), iteration_name="absent", top_n=1, gap_count=1)
    with pytest.raises(ProfileTraceError, match="gap count"):
        analyze_trace(
            two_iterations(), iteration_name="iteration", top_n=1, gap_count=0
        )
