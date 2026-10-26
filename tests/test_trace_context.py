from dataclasses import FrozenInstanceError

import pytest

from ai_journey.trace_context import TraceContext, child_context, parse_traceparent


TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN = "00f067aa0ba902b7"
HEADER = f"00-{TRACE}-{SPAN}-01"


def test_w3c_public_example_round_trips_and_masks_reserved_flags():
    context = parse_traceparent(HEADER)
    assert context == TraceContext(TRACE, SPAN, True)
    assert context.to_header() == HEADER
    assert parse_traceparent(f"00-{TRACE}-{SPAN}-09").sampled
    assert parse_traceparent(f"00-{TRACE}-{SPAN}-02").to_header().endswith("-00")
    assert parse_traceparent(f" \t{HEADER}\t ") == context
    with pytest.raises(FrozenInstanceError):
        context.trace_id = "1" * 32


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        b"00-public",
        42,
        HEADER.upper(),
        HEADER.replace(TRACE, "0" * 32),
        HEADER.replace(SPAN, "0" * 16),
        HEADER.replace(TRACE, "g" * 32),
        HEADER[:-1],
        HEADER + "-extra",
        "ff" + HEADER[2:],
        HEADER + "," + HEADER,
        HEADER + "\r\nother: private",
        HEADER + "\x00",
        HEADER + "\n",
        "00-" + "a" * 510,
        "01" + HEADER[2:] + "not-a-delimiter",
    ],
)
def test_invalid_or_ambiguous_headers_are_ignored(header):
    assert parse_traceparent(header) is None


def test_future_version_preserves_known_ids_without_retaining_extensions():
    context = parse_traceparent("01" + HEADER[2:] + "-opaque-vendor-value")
    assert context == TraceContext(TRACE, SPAN, True)
    assert context.to_header() == HEADER
    assert "opaque" not in repr(context)
    assert parse_traceparent("fe" + HEADER[2:]) == context


def test_children_preserve_trace_and_record_distinct_parent_relationship():
    parent = parse_traceparent(HEADER)
    children = [child_context(parent) for _ in range(32)]
    assert len({child.span_id for child in children}) == len(children)
    assert all(
        child.trace_id == TRACE and child.parent_span_id == SPAN and child.sampled
        for child in children
    )
    assert all(child.span_id != SPAN for child in children)
    next_hop = parse_traceparent(children[0].to_header())
    assert next_hop.trace_id == TRACE and next_hop.span_id == children[0].span_id
    assert next_hop.parent_span_id is None
    assert not child_context(parent, sampled=False).sampled


def test_new_traces_use_random_ids_and_do_not_claim_sampling():
    contexts = [child_context() for _ in range(32)]
    assert len({context.trace_id for context in contexts}) == len(contexts)
    assert all(
        not context.sampled and context.parent_span_id is None for context in contexts
    )
    assert all(
        parse_traceparent(context.to_header()) == context for context in contexts
    )


@pytest.mark.parametrize(
    "values",
    [
        dict(trace_id="0" * 32, span_id=SPAN),
        dict(trace_id=TRACE, span_id="0" * 16),
        dict(trace_id=TRACE.upper(), span_id=SPAN),
        dict(trace_id=TRACE, span_id=SPAN, sampled=1),
        dict(trace_id=TRACE, span_id=SPAN, parent_span_id=SPAN),
        dict(trace_id=TRACE, span_id=SPAN, parent_span_id="private-context"),
    ],
)
def test_constructed_contexts_obey_the_same_identifier_contract(values):
    with pytest.raises(ValueError):
        TraceContext(**values)


def test_broken_entropy_is_bounded_and_never_emits_zero_or_parent_ids(monkeypatch):
    parent = parse_traceparent(HEADER)
    values = iter(["0" * 16, SPAN, "1" * 16])
    monkeypatch.setattr(
        "ai_journey.trace_context.secrets.token_hex", lambda count: next(values)
    )
    assert child_context(parent).span_id == "1" * 16
    calls = []

    def zeros(count):
        calls.append(count)
        return "0" * (count * 2)

    monkeypatch.setattr("ai_journey.trace_context.secrets.token_hex", zeros)
    with pytest.raises(RuntimeError, match="distinct nonzero"):
        child_context()
    assert calls == [16] * 4


def test_invalid_child_controls_are_rejected_before_generating_ids(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("random generation must not run")

    monkeypatch.setattr("ai_journey.trace_context.secrets.token_hex", unexpected)
    with pytest.raises(TypeError, match="parent"):
        child_context(HEADER)
    with pytest.raises(ValueError, match="sampled"):
        child_context(sampled=1)
