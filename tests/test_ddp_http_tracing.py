import http.client

import pytest

from ai_journey.trace_context import parse_traceparent
from test_ddp_http_reliability import RecordingService, serving


PARENT = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"


@pytest.mark.parametrize("route", ["/health", "/unknown", "/generate", "/stream"])
def test_real_http_responses_propagate_safe_child_contexts(route):
    with serving(RecordingService()) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        client.request(
            "GET", route, headers={"traceparent": PARENT, "baggage": "secret=data"}
        )
        response = client.getresponse()
        context = parse_traceparent(response.getheader("traceparent"))
        assert context is not None
        assert context.trace_id == parse_traceparent(PARENT).trace_id
        assert context.span_id != parse_traceparent(PARENT).span_id
        assert context.sampled
        assert response.getheader("baggage") is None
        response.read()
        client.close()


@pytest.mark.parametrize(
    "header", [None, "invalid", "00-" + "0" * 32 + "-" + "0" * 16 + "-00"]
)
def test_untrusted_context_is_ignored_without_rejecting_the_request(header):
    with serving(RecordingService()) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        client.request(
            "GET", "/health", headers={"traceparent": header} if header else {}
        )
        response = client.getresponse()
        assert response.status == 200
        assert parse_traceparent(response.getheader("traceparent")) is not None
        response.read()
        client.close()


def test_duplicate_trace_contexts_do_not_select_an_ambiguous_parent():
    with serving(RecordingService()) as server:
        client = http.client.HTTPConnection(*server.server_address, timeout=2)
        client.putrequest("GET", "/health")
        client.putheader("traceparent", PARENT)
        client.putheader("traceparent", PARENT)
        client.endheaders()
        response = client.getresponse()
        context = parse_traceparent(response.getheader("traceparent"))
        assert context.trace_id != parse_traceparent(PARENT).trace_id
        response.read()
        client.close()
