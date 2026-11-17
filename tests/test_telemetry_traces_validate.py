import copy

import pytest

from ai_journey.telemetry_traces_otlp import (
    ResourceTelemetry,
    SpanEvent,
    TelemetrySpan,
    encode_export_request,
)
from ai_journey.telemetry_traces_validate import (
    OtlpValidationError,
    validate_export_request,
)
from ai_journey.trace_context import TraceContext

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
ROOT = TraceContext(TRACE, "00f067aa0ba902b7", True)
LIFETIME = TraceContext(TRACE, "1111111111111111", True, ROOT.span_id)
ATTEMPT = TraceContext(TRACE, "2222222222222222", True, ROOT.span_id)
LABEL = {
    "deployment.environment.name": "local-cpu-lab",
    "ai_journey.evidence.boundary": "Local CPU lab telemetry only.",
}


def make_span(context, start, end, **changes):
    values = dict(
        context=context,
        name="span",
        kind="internal",
        start_unix_nano=start,
        end_unix_nano=end,
        attributes={"ai_journey.lab": "serving-recovery"},
    )
    return TelemetrySpan(**(values | changes))


def sample_request():
    fault = SpanEvent("ai_journey.lifecycle.fault", 30, {"process.pid": 7})
    serving = ResourceTelemetry(
        {"service.name": "serving"} | LABEL,
        "ai_journey.telemetry_traces",
        "1",
        (
            make_span(ROOT, 10, 100, name="run"),
            make_span(
                LIFETIME,
                10,
                40,
                name="lifetime",
                events=(SpanEvent("ready", 20), fault),
                status="error",
                status_message="SIGKILL injected",
            ),
            make_span(ATTEMPT, 25, 35, name="POST /generate", kind="client"),
        ),
    )
    worker = ResourceTelemetry(
        {"service.name": "worker"} | LABEL,
        "ai_journey.telemetry_traces",
        "1",
        (make_span(TraceContext("a" * 32, "b" * 16, True), 5, 6, name="worker"),),
    )
    return encode_export_request([serving, worker])


def spans(request, resource=0):
    return request["resourceSpans"][resource]["scopeSpans"][0]["spans"]


def test_encoded_requests_validate_and_summarize():
    summary = validate_export_request(sample_request())

    assert summary.to_mapping() == {
        "resources": 2,
        "trace_ids": sorted([TRACE, "a" * 32]),
        "spans": 4,
        "events": 2,
        "spans_by_kind": {"client": 1, "internal": 3},
        "error_spans": 1,
    }


def set_field(index, name, value):
    return lambda request: spans(request)[index].__setitem__(name, value)


def set_resource(key, value):
    def mutate(request):
        attributes = request["resourceSpans"][0]["resource"]["attributes"]
        for entry in attributes:
            if entry["key"] == key:
                entry["value"] = value
                return
        attributes.append({"key": key, "value": value})

    return mutate


def drop_resource(key):
    def mutate(request):
        attributes = request["resourceSpans"][0]["resource"]["attributes"]
        attributes[:] = [entry for entry in attributes if entry["key"] != key]

    return mutate


def swap(first, second):
    def mutate(request):
        items = spans(request)
        items[first], items[second] = items[second], items[first]

    return mutate


def copy_worker_span_into_serving(request):
    moved = copy.deepcopy(spans(request, 1)[0])
    moved.update(startTimeUnixNano="50", endTimeUnixNano="60")
    spans(request).append(moved)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (set_field(0, "traceId", TRACE.upper()), "traceId must be 32"),
        (set_field(0, "traceId", "0" * 32), "traceId must be 32"),
        (set_field(1, "spanId", "1" * 15), "spanId must be 16"),
        (set_field(1, "parentSpanId", "3" * 16), "names no earlier span"),
        (set_field(1, "parentSpanId", "1" * 16), "its own parent"),
        (swap(0, 1), "names no earlier span"),
        (set_field(2, "startTimeUnixNano", 25), "decimal string"),
        (set_field(2, "startTimeUnixNano", "025"), "decimal string"),
        (set_field(2, "startTimeUnixNano", "0"), "unsigned 64-bit"),
        (set_field(0, "endTimeUnixNano", str(2**64)), "unsigned 64-bit"),
        (set_field(2, "endTimeUnixNano", "24"), "ends before it starts"),
        (set_field(2, "endTimeUnixNano", "101"), "outside its parent"),
        (set_field(2, "startTimeUnixNano", "5"), "outside its parent"),
        (set_field(1, "spanId", "2" * 16), "repeats span"),
        (lambda r: spans(r)[2].pop("parentSpanId"), "second root"),
        (set_field(2, "kind", 0), "SpanKind"),
        (set_field(2, "kind", True), "SpanKind"),
        (set_field(2, "kind", "SPAN_KIND_CLIENT"), "SpanKind"),
        (set_field(2, "name", ""), "printable ASCII"),
        (set_field(2, "links", []), "unknown field links"),
        (lambda r: spans(r)[2].pop("status"), "lacks status"),
        (set_field(2, "status", {"code": 0, "message": "x"}), "without an error"),
        (set_field(2, "status", {"code": 3}), "0, 1 or 2"),
        (set_field(1, "events", []), "list of 1 to"),
        (lambda r: spans(r)[1]["events"].reverse(), "out of order"),
        (
            lambda r: spans(r)[1]["events"][1].__setitem__("timeUnixNano", "41"),
            "outside its span",
        ),
        (
            lambda r: spans(r)[1]["attributes"].append(spans(r)[1]["attributes"][0]),
            "repeats key",
        ),
        (
            set_field(2, "attributes", [{"key": "Bad", "value": {"boolValue": True}}]),
            "lowercase dotted",
        ),
        (
            set_field(2, "attributes", [{"key": "a.n", "value": {"intValue": 1}}]),
            "64-bit integer",
        ),
        (
            set_field(
                2, "attributes", [{"key": "a.n", "value": {"intValue": str(2**63)}}]
            ),
            "64-bit integer",
        ),
        (
            set_field(2, "attributes", [{"key": "a.d", "value": {"doubleValue": 1.5}}]),
            "unsupported value type",
        ),
        (
            set_field(
                2,
                "attributes",
                [{"key": "a.b", "value": {"boolValue": True, "intValue": "1"}}],
            ),
            "exactly one typed value",
        ),
        (
            lambda r: spans(r).sort(key=lambda s: -int(s["startTimeUnixNano"])),
            "earlier",
        ),
        (copy_worker_span_into_serving, "across scopes"),
        (drop_resource("service.name"), "service.name"),
        (
            set_resource("deployment.environment.name", {"stringValue": "production"}),
            "local-cpu-lab",
        ),
        (drop_resource("ai_journey.evidence.boundary"), "boundary"),
        (
            lambda r: r["resourceSpans"][0]["scopeSpans"][0].__setitem__("spans", []),
            "list of 1",
        ),
        (lambda r: r.__setitem__("resourceSpans", []), "list of 1"),
        (lambda r: r.__setitem__("partialSuccess", {}), "unknown field partialSuccess"),
        (
            lambda r: r["resourceSpans"][0].__setitem__("schemaUrl", "x"),
            "unknown field schemaUrl",
        ),
    ],
)
def test_structural_violations_are_rejected_with_their_location(mutate, message):
    request = copy.deepcopy(sample_request())
    mutate(request)
    with pytest.raises(OtlpValidationError, match=message):
        validate_export_request(request)


def test_spans_in_a_scope_must_be_ordered_by_start_time():
    request = copy.deepcopy(sample_request())
    # Siblings listed late-first still have their parent earlier.
    swap(1, 2)(request)
    with pytest.raises(OtlpValidationError, match="ordered by start time"):
        validate_export_request(request)


def test_non_objects_are_rejected_without_type_errors():
    for payload in (None, [], "request", {"resourceSpans": [None]}):
        with pytest.raises(OtlpValidationError, match="JSON object|list of 1"):
            validate_export_request(payload)
