import json
from hashlib import sha256

import pytest

from ai_journey.telemetry_traces_io import TelemetryError
from ai_journey.telemetry_traces_ids import (
    derive_span_id,
    derive_trace_id,
    span_context,
)
from ai_journey.trace_context import TraceContext, parse_traceparent

RUN = "day-70-serving-recovery-c5c20a75744245aa"
HEAD = "f23bfceac28709ebacaa307150875d2a1f7541c472f3f9bf621bbb0d5e089d5a"


def digest(payload):
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_trace_ids_are_sha256_prefixes_of_the_lab_run_and_trace_head():
    trace_id = derive_trace_id(
        lab="serving-recovery", run_id=RUN, trace_head_sha256=HEAD
    )

    expected = digest(
        {
            "kind": "ai-journey-telemetry-trace",
            "lab": "serving-recovery",
            "run_id": RUN,
            "trace_head_sha256": HEAD,
        }
    )[:32]
    assert trace_id == expected
    assert trace_id == derive_trace_id(
        lab="serving-recovery", run_id=RUN, trace_head_sha256=HEAD
    )
    others = {
        derive_trace_id(lab="worker-recovery", run_id=RUN, trace_head_sha256=HEAD),
        derive_trace_id(lab="serving-recovery", run_id=RUN, trace_head_sha256="1" * 64),
    }
    assert trace_id not in others and len(others) == 2


def test_span_ids_are_stable_per_key_and_rendered_as_sampled_w3c_contexts():
    trace_id = derive_trace_id(
        lab="worker-recovery", run_id=RUN, trace_head_sha256=HEAD
    )
    root = span_context(trace_id, "run")
    key = "worker-attempt/attempt-1-0123456789abcdef"
    child = span_context(trace_id, key, parent=root)

    assert (
        root.span_id
        == digest(
            {
                "kind": "ai-journey-telemetry-span",
                "span_key": "run",
                "trace_id": trace_id,
            }
        )[:16]
    )
    assert root == TraceContext(trace_id, root.span_id, True, None)
    assert child.parent_span_id == root.span_id and child.trace_id == trace_id
    assert child == span_context(trace_id, key, parent=root)
    header = child.to_header()
    # Sampled, and the W3C Level 2 random flag (0x02) stays clear.
    assert header == f"00-{trace_id}-{child.span_id}-01"
    assert parse_traceparent(header) == TraceContext(trace_id, child.span_id, True)
    keys = [f"request/req-{index:04d}" for index in range(64)]
    assert len({derive_span_id(trace_id, key) for key in keys}) == len(keys)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (dict(lab="training", run_id=RUN, trace_head_sha256=HEAD), "lab"),
        (dict(lab="worker-recovery", run_id="Run-0", trace_head_sha256=HEAD), "run_id"),
        (
            dict(lab="worker-recovery", run_id=RUN, trace_head_sha256=HEAD.upper()),
            "SHA",
        ),
        (dict(lab="worker-recovery", run_id=RUN, trace_head_sha256=None), "SHA"),
    ],
)
def test_trace_ids_need_a_known_lab_run_identity_and_head_digest(values, message):
    with pytest.raises(TelemetryError, match=message):
        derive_trace_id(**values)


@pytest.mark.parametrize(
    ("trace_id", "key", "parent", "message"),
    [
        ("0" * 32, "run", None, "trace_id"),
        ("A" * 32, "run", None, "trace_id"),
        ("a" * 31, "run", None, "trace_id"),
        ("a" * 32, "Run", None, "span key"),
        ("a" * 32, "", None, "span key"),
        ("a" * 32, "x" * 257, None, "span key"),
        ("a" * 32, "run", "0" * 16, "parent_span_id"),
    ],
)
def test_span_ids_reject_malformed_inputs(trace_id, key, parent, message):
    with pytest.raises(TelemetryError, match=message):
        derive_span_id(trace_id, key, parent_span_id=parent)


def test_parents_must_be_contexts_of_the_same_trace():
    root = span_context("a" * 32, "run")
    with pytest.raises(TelemetryError, match="same trace"):
        span_context("b" * 32, "phase/detection", parent=root)
    with pytest.raises(TypeError, match="TraceContext"):
        span_context("a" * 32, "phase/detection", parent=root.to_header())


def test_zero_or_parent_equal_digests_are_rederived_within_a_bound(monkeypatch):
    parent = "1" * 16
    values = iter(["0" * 64, parent + "0" * 48, "2" * 64])
    payloads = []

    def fake(payload):
        payloads.append(payload)
        return next(values)

    monkeypatch.setattr("ai_journey.telemetry_traces_ids.canonical_sha256", fake)
    assert derive_span_id("a" * 32, "run", parent_span_id=parent) == "2" * 16
    assert [payload.get("salt") for payload in payloads] == [None, 1, 2]

    monkeypatch.setattr(
        "ai_journey.telemetry_traces_ids.canonical_sha256",
        lambda payload: payloads.append(payload) or "0" * 64,
    )
    payloads.clear()
    with pytest.raises(TelemetryError, match="distinct nonzero"):
        derive_trace_id(lab="serving-recovery", run_id=RUN, trace_head_sha256=HEAD)
    assert len(payloads) == 4
