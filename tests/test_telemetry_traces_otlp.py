import pytest

from ai_journey.telemetry_traces_io import TelemetryError
from ai_journey.telemetry_traces_otlp import (
    MonotonicEpochMapping,
    ResourceTelemetry,
    SpanEvent,
    TelemetrySpan,
    encode_any_value,
    encode_export_request,
    encode_span,
)
from ai_journey.trace_context import TraceContext

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
ROOT = TraceContext(TRACE, "00f067aa0ba902b7", True)
ANCHOR = 1_794_787_200_000_000_000


def child(span_id, parent=ROOT):
    return TraceContext(parent.trace_id, span_id, True, parent.span_id)


def span(context, start, end, *, name="span", kind="internal", **changes):
    values = {
        "context": context,
        "name": name,
        "kind": kind,
        "start_unix_nano": start,
        "end_unix_nano": end,
        "attributes": {"ai_journey.lab": "serving-recovery"},
    }
    return TelemetrySpan(**(values | changes))


def resource(*spans, **attributes):
    return ResourceTelemetry(
        attributes={"service.name": "ai-journey-test"} | attributes,
        scope_name="ai_journey.telemetry_traces",
        scope_version="1",
        spans=spans,
    )


def test_monotonic_mapping_keeps_exact_differences_from_the_declared_anchor():
    mapping = MonotonicEpochMapping(
        origin_monotonic_ns=79491133515875, anchor_unix_nano=ANCHOR
    )

    assert mapping.to_unix_nano(79491133515875) == ANCHOR
    later = mapping.to_unix_nano(79491222070833)
    assert later - ANCHOR == 79491222070833 - 79491133515875
    assert mapping.to_mapping() == {
        "clock": "time.monotonic_ns in the experiment process on one host",
        "formula": (
            "unix_nano = anchor_unix_nano + (monotonic_ns - origin_monotonic_ns)"
        ),
        "origin_monotonic_ns": 79491133515875,
        "anchor_unix_nano": ANCHOR,
    }
    with pytest.raises(TelemetryError, match="precedes the mapping origin"):
        mapping.to_unix_nano(79491133515874)
    with pytest.raises(TelemetryError, match="integer"):
        mapping.to_unix_nano(True)
    edge = MonotonicEpochMapping(origin_monotonic_ns=0, anchor_unix_nano=2**64 - 1)
    with pytest.raises(TelemetryError, match="overflows"):
        edge.to_unix_nano(1)


@pytest.mark.parametrize(
    ("origin", "anchor"),
    [(-1, ANCHOR), (2**63, ANCHOR), (0, 0), (0, 2**64), (0, float(ANCHOR)), (False, 1)],
)
def test_mapping_bounds_are_explicit(origin, anchor):
    with pytest.raises(TelemetryError):
        MonotonicEpochMapping(origin_monotonic_ns=origin, anchor_unix_nano=anchor)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "batch"}, "span kind"),
        ({"name": ""}, "printable ASCII"),
        ({"name": " padded"}, "printable ASCII"),
        ({"start_unix_nano": 0}, "span start"),
        ({"end_unix_nano": 5}, "ends before"),
        ({"attributes": {"Service.Name": "x"}}, "lowercase dotted"),
        ({"attributes": {"ai_journey..lab": "x"}}, "lowercase dotted"),
        ({"attributes": {"ai_journey.ratio": 0.5}}, "string, boolean or integer"),
        ({"attributes": {"ai_journey.items": 2**63}}, "64-bit"),
        ({"attributes": {"ai_journey.note": "line\nbreak"}}, "printable ASCII"),
        ({"attributes": {f"a.k{i}": i for i in range(129)}}, "128 attributes"),
        ({"events": (SpanEvent("late", 30),)}, "ordered inside"),
        ({"events": (SpanEvent("b", 15), SpanEvent("a", 12))}, "ordered inside"),
        ({"events": [SpanEvent("a", 12)]}, "tuple of SpanEvent"),
        ({"status": "failed"}, "unset, ok or error"),
        ({"status_message": "boom"}, "only an error status"),
    ],
)
def test_spans_validate_names_kinds_times_attributes_events_and_status(
    changes, message
):
    with pytest.raises((TelemetryError, TypeError), match=message):
        span(ROOT, 10, 20, **changes)


def test_resources_hold_complete_contained_traces():
    first = child("1111111111111111")
    with pytest.raises(TelemetryError, match="service.name"):
        ResourceTelemetry({"ai_journey.lab": "x"}, "scope", "1", (span(ROOT, 1, 9),))
    with pytest.raises(TelemetryError, match="missing parent"):
        resource(span(first, 2, 3))
    with pytest.raises(TelemetryError, match="unique"):
        resource(span(ROOT, 1, 9), span(ROOT, 1, 9))
    with pytest.raises(TelemetryError, match="outside its parent"):
        resource(span(ROOT, 1, 9), span(first, 2, 10))
    other_root = TraceContext(TRACE, "2222222222222222", True)
    with pytest.raises(TelemetryError, match="exactly one root"):
        resource(span(ROOT, 1, 9), span(other_root, 1, 9))
    a = TraceContext(TRACE, "3333333333333333", True, "4444444444444444")
    b = TraceContext(TRACE, "4444444444444444", True, "3333333333333333")
    with pytest.raises(TelemetryError, match="cycle"):
        resource(span(ROOT, 1, 9), span(a, 2, 3), span(b, 2, 3))


def test_spans_encode_as_otlp_json_with_hex_ids_string_times_and_int_enums():
    fault = SpanEvent(
        "ai_journey.lifecycle.fault",
        15,
        {"ai_journey.fault.signal": "SIGKILL", "process.pid": 42064},
    )
    encoded = encode_span(
        span(
            child("1111111111111111"),
            10,
            20,
            name="server lifetime 1",
            events=(fault,),
            status="error",
            status_message="SIGKILL injected",
            attributes={"process.pid": 42064, "ai_journey.lab": "x", "a.b": True},
        )
    )

    assert encoded == {
        "traceId": TRACE,
        "spanId": "1111111111111111",
        "parentSpanId": ROOT.span_id,
        "name": "server lifetime 1",
        "kind": 1,
        "startTimeUnixNano": "10",
        "endTimeUnixNano": "20",
        "attributes": [
            {"key": "a.b", "value": {"boolValue": True}},
            {"key": "ai_journey.lab", "value": {"stringValue": "x"}},
            {"key": "process.pid", "value": {"intValue": "42064"}},
        ],
        "events": [
            {
                "timeUnixNano": "15",
                "name": "ai_journey.lifecycle.fault",
                "attributes": [
                    {
                        "key": "ai_journey.fault.signal",
                        "value": {"stringValue": "SIGKILL"},
                    },
                    {"key": "process.pid", "value": {"intValue": "42064"}},
                ],
            }
        ],
        "status": {"code": 2, "message": "SIGKILL injected"},
    }
    root = encode_span(span(ROOT, 1, 30, kind="client"))
    assert "parentSpanId" not in root and "events" not in root
    assert (root["kind"], root["status"]) == (3, {"code": 0})
    assert encode_any_value(-(2**63)) == {"intValue": str(-(2**63))}
    with pytest.raises(TelemetryError, match="attribute values"):
        encode_any_value(1.5)


def test_export_requests_order_spans_by_start_then_depth_per_resource():
    first = child("1111111111111111")
    grandchild = child("2222222222222222", parent=first)
    late = child("3333333333333333")
    run = resource(
        span(late, 7, 8, name="late"),
        span(grandchild, 2, 3, name="grandchild"),
        span(first, 2, 5, name="first"),
        span(ROOT, 2, 9, name="root"),
    )
    other = ResourceTelemetry(
        {"service.name": "other"},
        "ai_journey.telemetry_traces",
        "1",
        (span(TraceContext("a" * 32, "b" * 16, True), 1, 2, name="other root"),),
    )

    request = encode_export_request([run, other])

    resource_spans = request["resourceSpans"]
    assert [entry["resource"] for entry in resource_spans] == [
        {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "ai-journey-test"}}
            ]
        },
        {"attributes": [{"key": "service.name", "value": {"stringValue": "other"}}]},
    ]
    scope = resource_spans[0]["scopeSpans"][0]
    assert scope["scope"] == {"name": "ai_journey.telemetry_traces", "version": "1"}
    assert [entry["name"] for entry in scope["spans"]] == [
        "root",
        "first",
        "grandchild",
        "late",
    ]
    with pytest.raises(TelemetryError, match="split across resources"):
        encode_export_request([run, run])
    with pytest.raises(TelemetryError, match="at least one"):
        encode_export_request([])
    with pytest.raises(TypeError, match="ResourceTelemetry"):
        encode_export_request([run.spans[0]])
