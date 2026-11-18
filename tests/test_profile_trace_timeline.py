import pytest

from ai_journey.profile_trace_events import ProfileTraceError, parse_trace
from ai_journey.profile_trace_timeline import MAX_WIDTH, MIN_WIDTH, render_timeline
from test_profile_trace_analysis import two_iterations
from test_profile_trace_events import PID, TID, document, span


def test_the_two_iterations_draw_as_rows_per_depth_with_a_legend():
    text = render_timeline(two_iterations(), width=18, depth_limit=3)

    assert text == (
        "timeline extent_ns=180000 columns=18 ns_per_column=10000.0\n"
        f"thread pid={PID} tid={TID}\n"
        "  d0 |AAAAAAAAAA...AAAAA|\n"
        "  d1 |BBBDDDDDD....BDDDD|\n"
        "  d2 |.C................|\n"
        "legend\n"
        "  A iteration\n"
        "  B side/dispatch\n"
        "  C aten::mm\n"
        "  D side/synchronize\n"
    )


def test_the_depth_limit_drops_deeper_rows_and_their_legend():
    text = render_timeline(two_iterations(), width=MIN_WIDTH, depth_limit=1)
    assert "d1" not in text and "side/dispatch" not in text
    assert text.splitlines()[-1] == "  A iteration"


def test_zero_length_spans_and_threads_each_get_a_mark():
    trace = parse_trace(
        document(span("a", 0, 16), span("mark", 8, 0, tid=5), span("b", 15, 1))
    )
    lines = render_timeline(trace, width=16, depth_limit=1).splitlines()
    # b nests inside a, so a depth limit of 1 leaves it out.
    assert lines[2] == "  d0 |AAAAAAAAAAAAAAAA|"
    assert lines[4] == "  d0 |........B.......|"
    assert lines[-3:] == ["legend", "  A a", "  B mark"]


def test_the_longest_overlap_wins_a_column_and_ties_go_to_the_earlier_span():
    trace = parse_trace(document(span("a", 0, 3), span("b", 3, 29)))
    # Each column is 2 us; column 1 holds 1 us of a and 1 us of b.
    row = render_timeline(trace, width=16, depth_limit=1).splitlines()[2]
    assert row == "  d0 |AABBBBBBBBBBBBBB|"


@pytest.mark.parametrize(
    "width, depth",
    [(MIN_WIDTH - 1, 1), (MAX_WIDTH + 1, 1), (16, 0), (16, 9), (True, 1)],
)
def test_bounds_are_checked(width, depth):
    with pytest.raises(ProfileTraceError):
        render_timeline(two_iterations(), width=width, depth_limit=depth)


def test_the_timeline_needs_a_parsed_trace():
    with pytest.raises(TypeError):
        render_timeline({}, width=16, depth_limit=1)
