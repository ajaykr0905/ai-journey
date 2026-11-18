import json
import os
from decimal import Decimal

import pytest

from ai_journey.profile_trace_events import (
    MAX_EVENTS,
    MAX_TRACE_BYTES,
    ParsedTrace,
    ProfileTraceError,
    TraceFieldError,
    TraceNestingError,
    TracePhaseError,
    TraceSizeError,
    TraceSyntaxError,
    TraceTimeError,
    decode_trace_json,
    load_trace,
    microseconds_to_ns,
    parse_trace,
    parse_trace_bytes,
)

PID = 33879
TID = 43692759

# The layout torch 2.15.0 writes on this host, cut to one event of each kind.
TORCH_LAYOUT = b"""{
    "schemaVersion": 1,
    "deviceProperties": [],
    "trace_id": "F06FF7130E3B4AB6B0F966B97AC019E8",
    "record_shapes": 1,
    "host_name": "redacted-host",
    "displayTimeUnit": "ms",
    "baseTimeNanoseconds": 1790857026000000000,
    "traceEvents": [
  {
    "ph": "M", "name": "thread_name", "ts": 4112641660692.000,
    "pid": 33879, "tid": 43692759, "args": {"name": "thread 43692759 ()"}
  },
  {
    "ph": "X", "cat": "user_annotation", "name": "iter_0",
    "pid": 33879, "tid": 43692759, "ts": 4112641662268.470, "dur": 7532.139
    ,
    "args": {"External id": 1, "Record function id": 0, "Ev Idx": 0}
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::mm",
    "pid": 33879, "tid": 43692759, "ts": 4112641669344.188, "dur": 214.868
    ,
    "args": {"Input Dims": [[256, 256], [256, 256]], "Ev Idx": 5}
  },
  {
    "ph": "X", "cat": "Trace", "name": "PyTorch Profiler (0)",
    "pid": "Spans", "tid": "PyTorch Profiler",
    "ts": 4112641660101.000, "dur": 13824.000, "args": {"Op count": 0}
  },
  {
    "ph": "M", "name": "process_sort_index", "ts": 4112641660101.000,
    "pid": "Spans", "tid": 0, "args": {"sort_index": 536870912}
  },
  {
    "ph": "i", "s": "g", "name": "Record Window End",
    "pid": "", "tid": "", "ts": 4112641674955.000
  }
  ],"traceName": "trace.json" }"""


def span(name, ts, dur, *, tid=TID, cat="user_annotation"):
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": PID,
        "tid": tid,
        "ts": ts,
        "dur": dur,
    }


def document(*events, **fields):
    return {"traceEvents": list(events), **fields}


def encode(value):
    return json.dumps(value).encode("utf-8")


def test_the_torch_layout_parses_into_exact_nanoseconds():
    trace = parse_trace_bytes(TORCH_LAYOUT)

    assert isinstance(trace, ParsedTrace)
    assert trace.event_count == 6
    assert trace.display_time_unit == "ms"
    first, inner, window = trace.spans
    assert (first.name, first.category, first.pid, first.tid) == (
        "iter_0",
        "user_annotation",
        PID,
        TID,
    )
    assert first.start_ns == 4112641662268470
    assert first.duration_ns == 7532139
    assert first.end_ns == 4112641669800609
    assert inner.thread == (PID, TID)
    assert window.pid == "Spans" and window.tid == "PyTorch Profiler"
    (instant,) = trace.instants
    assert (instant.name, instant.scope, instant.pid) == ("Record Window End", "g", "")
    assert trace.thread_names() == {(PID, TID): "thread 43692759 ()"}
    assert [item.name for item in trace.metadata] == [
        "thread_name",
        "process_sort_index",
    ]


def test_microseconds_convert_exactly_and_refuse_fractions_of_a_nanosecond():
    assert microseconds_to_ns(Decimal("4112641662268.470"), "ts") == 4112641662268470
    assert microseconds_to_ns(7, "ts") == 7000
    assert microseconds_to_ns(Decimal("0.001"), "ts") == 1
    for value in (Decimal("0.0005"), Decimal("1.2345")):
        with pytest.raises(TraceTimeError, match="whole number of nanoseconds"):
            microseconds_to_ns(value, "ts")
    with pytest.raises(TraceTimeError, match="from 0"):
        microseconds_to_ns(-1, "ts")
    with pytest.raises(TraceTimeError, match="from 0"):
        microseconds_to_ns(10**20, "ts")
    for value in (True, "1", 1.5, None):
        with pytest.raises(TraceFieldError, match="JSON number"):
            microseconds_to_ns(value, "ts")


def test_begin_and_end_pairs_become_spans_in_stack_order():
    trace = parse_trace(
        document(
            {"ph": "B", "name": "outer", "pid": 1, "tid": 2, "ts": 10, "cat": "c"},
            {"ph": "B", "name": "inner", "pid": 1, "tid": 2, "ts": 11},
            {"ph": "E", "pid": 1, "tid": 2, "ts": 12},
            {"ph": "E", "name": "outer", "pid": 1, "tid": 2, "ts": 20},
        )
    )

    inner, outer = trace.spans
    assert (inner.name, inner.start_ns, inner.duration_ns) == ("inner", 11000, 1000)
    assert (outer.name, outer.category, outer.duration_ns) == ("outer", "c", 10000)


@pytest.mark.parametrize(
    "events, message",
    [
        ([{"ph": "E", "pid": 1, "tid": 2, "ts": 1}], "no open B"),
        ([{"ph": "B", "name": "a", "pid": 1, "tid": 2, "ts": 1}], "has no E"),
        (
            [
                {"ph": "B", "name": "a", "pid": 1, "tid": 2, "ts": 1},
                {"ph": "E", "name": "b", "pid": 1, "tid": 2, "ts": 2},
            ],
            "closes 'b'",
        ),
        (
            [
                {"ph": "B", "name": "a", "pid": 1, "tid": 2, "ts": 5},
                {"ph": "E", "pid": 1, "tid": 2, "ts": 4},
            ],
            "before its B",
        ),
        (
            [
                {"ph": "B", "name": "a", "pid": 1, "tid": 2, "ts": 1},
                {"ph": "E", "pid": 1, "tid": 3, "ts": 2},
            ],
            "no open B",
        ),
    ],
)
def test_unpaired_or_mismatched_begin_and_end_events_are_refused(events, message):
    with pytest.raises(TraceNestingError, match=message):
        parse_trace(document(*events))


@pytest.mark.parametrize("phase", ["s", "f", "C", "b", "I", None, 1])
def test_phases_outside_the_torch_set_are_refused_by_name(phase):
    event = span("a", 1, 1)
    event["ph"] = phase
    with pytest.raises(TracePhaseError, match="phase"):
        parse_trace(document(event))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"dur": None}, "dur must be a JSON number"),
        ({"ts": "1"}, "ts must be a JSON number"),
        ({"pid": True}, "pid must be an integer or a string"),
        ({"tid": -1}, "tid must be a non-negative"),
        ({"name": 3}, "name must be a string"),
        ({"name": "x" * 513}, "name must be a string"),
        ({"cat": None}, "cat must be a string"),
        ({"args": []}, "args must be a JSON object"),
        ({"extra": 1}, "unknown extra"),
    ],
)
def test_complete_events_check_every_field_type(change, message):
    with pytest.raises(TraceFieldError, match=message):
        parse_trace(document({**span("a", 1, 1), **change}))


def test_missing_fields_are_named():
    event = span("a", 1, 1)
    del event["dur"]
    del event["tid"]
    with pytest.raises(TraceFieldError, match="missing dur, tid"):
        parse_trace(document(event))


def test_instants_and_metadata_are_checked():
    instant = {"ph": "i", "name": "mark", "pid": 1, "tid": 1, "ts": 1, "s": "x"}
    with pytest.raises(TraceFieldError, match="scope"):
        parse_trace(document(instant))
    del instant["s"]
    assert parse_trace(document(instant)).instants[0].scope == "t"
    meta = {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": 5}}
    with pytest.raises(TraceFieldError, match="hold only name as str"):
        parse_trace(document(meta))
    meta["args"] = {"name": "main", "other": 1}
    with pytest.raises(TraceFieldError, match="hold only name"):
        parse_trace(document(meta))
    meta.update(name="thread_color", args={"name": "red"})
    with pytest.raises(TraceFieldError, match="metadata name"):
        parse_trace(document(meta))
    meta.update(name="thread_sort_index", args={"sort_index": True})
    with pytest.raises(TraceFieldError, match="sort_index as int"):
        parse_trace(document(meta))


def test_the_root_is_checked():
    with pytest.raises(TraceFieldError, match="no traceEvents"):
        parse_trace({})
    with pytest.raises(TraceFieldError, match="must be a JSON array"):
        parse_trace({"traceEvents": {}})
    with pytest.raises(TraceFieldError, match="unknown trace fields: other"):
        parse_trace(document(other=1))
    with pytest.raises(TraceFieldError, match="schemaVersion must be int"):
        parse_trace(document(schemaVersion="1"))
    with pytest.raises(TraceFieldError, match="displayTimeUnit"):
        parse_trace(document(displayTimeUnit="s"))
    with pytest.raises(TraceFieldError, match="event 0 must be a JSON object"):
        parse_trace(document(1))
    with pytest.raises(TraceFieldError, match="root must be a JSON object"):
        parse_trace_bytes(b"[]")


@pytest.mark.parametrize(
    "payload, message",
    [
        (b'{"traceEvents": [], "traceEvents": []}', "duplicate JSON field"),
        (b'{"traceEvents": [NaN]}', "non-finite"),
        (b'{"traceEvents": [Infinity]}', "non-finite"),
        (b'{"traceEvents": [', "strict UTF-8 JSON"),
        (b'\xff{"traceEvents": []}', "strict UTF-8 JSON"),
        (b'{"traceEvents": [' + b"1" * 5000 + b"]}", "strict UTF-8 JSON"),
    ],
)
def test_malformed_json_is_a_syntax_error(payload, message):
    with pytest.raises(TraceSyntaxError, match=message):
        decode_trace_json(payload)


def test_bounds_hold_for_bytes_and_events():
    with pytest.raises(TraceSizeError, match="exceeds"):
        decode_trace_json(b" " * (MAX_TRACE_BYTES + 1))
    marks = [{"ph": "i", "name": "m", "pid": 1, "tid": 1, "ts": 1}] * (MAX_EVENTS + 1)
    with pytest.raises(TraceSizeError, match="more than"):
        parse_trace(document(*marks))
    with pytest.raises(TypeError):
        decode_trace_json("{}")


def test_every_named_error_is_a_profile_trace_error():
    for kind in (
        TraceFieldError,
        TraceNestingError,
        TracePhaseError,
        TraceSizeError,
        TraceSyntaxError,
        TraceTimeError,
    ):
        assert issubclass(kind, ProfileTraceError)
    assert issubclass(ProfileTraceError, ValueError)


def test_files_are_read_with_the_bounded_regular_file_reader(tmp_path):
    path = tmp_path / "trace.json"
    path.write_bytes(TORCH_LAYOUT)
    assert load_trace(path).event_count == 6
    link = tmp_path / "link.json"
    os.symlink(path, link)
    with pytest.raises(ProfileTraceError, match="non-symlink"):
        load_trace(link)
    with pytest.raises(ProfileTraceError, match="regular"):
        load_trace(tmp_path)
    big = tmp_path / "big.json"
    with big.open("wb") as handle:
        handle.truncate(MAX_TRACE_BYTES + 1)
    with pytest.raises(TraceSizeError, match="exceeds"):
        load_trace(big)
