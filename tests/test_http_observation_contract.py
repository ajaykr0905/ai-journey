import copy
from unittest.mock import patch

import pytest

from ai_journey.http_load import HTTP_LOAD_SCOPE, http_load_totals
from ai_journey.http_load_profile import HTTPWorkloadProfile, http_request_schedule
from ai_journey.http_load_report import (
    build_http_load_report,
    http_latency_summary,
    http_report_digest,
    verify_http_load_report,
)


def baseline(*, streaming=True, cancellation=False, cancel_after_tokens=1):
    profile = HTTPWorkloadProfile(
        measured_requests=1,
        warmup_requests=0,
        concurrency=1,
        max_new_tokens=3,
        timeout_seconds=0.01,
        streaming=streaming,
        cancel_every=1 if cancellation else 0,
        cancel_after_tokens=cancel_after_tokens,
    )
    run_id = "a" * 32
    spec = http_request_schedule(profile, run_id)[0]
    observation = {
        "request_id": spec["request"]["request_id"],
        "route": "/stream" if streaming else "/generate",
        "started_ns": 1_000_000,
        "headers_ns": 1_500_000,
        "finished_ns": 5_000_000,
        "status": 200,
        "outcome": "completed",
        "tokens": [3, 4, 5],
        "token_received_ns": [2_000_000, 3_000_000, 4_000_000] if streaming else [],
        "model_sha256": "b" * 64,
        "finish_reason": "length",
        "error_kind": None,
    }
    if cancellation:
        observation.update(
            outcome="cancelled",
            tokens=[3, 4][:cancel_after_tokens],
            token_received_ns=[2_000_000, 3_000_000][:cancel_after_tokens],
            model_sha256=None,
            finish_reason=None,
            error_kind="client_closed_stream",
        )
    rows = [{"index": 0, "observation": observation}]
    return {
        "schema": "ai-journey-http-load-raw-v1",
        "scope": HTTP_LOAD_SCOPE,
        "created_at_utc": "2026-10-26T00:00:00+00:00",
        "endpoint": "http://127.0.0.1:12345",
        "run_id": run_id,
        "profile": profile.to_mapping(),
        "profile_sha256": profile.fingerprint(),
        "warmup": [],
        "measured": rows,
        "work_started_ns": 0,
        "work_finished_ns": 30_000_000,
        "totals": http_load_totals(
            rows, work_started_ns=0, work_finished_ns=30_000_000
        ),
    }


def forged_report(changes, **controls):
    raw = copy.deepcopy(baseline(**controls))
    raw["measured"][0]["observation"].update(changes)
    raw["totals"] = http_load_totals(
        raw["measured"],
        work_started_ns=raw["work_started_ns"],
        work_finished_ns=raw["work_finished_ns"],
    )
    # Model an adversary recomputing every derived field, rather than relying
    # on a stale digest or stale percentiles to expose an impossible record.
    with patch("ai_journey.http_load_report.validate_http_measurement"):
        report = build_http_load_report(raw)
        report["latencies"] = http_latency_summary(raw)
    report["report_sha256"] = http_report_digest(report)
    return report


@pytest.mark.parametrize(
    "changes,controls",
    [
        (
            {
                "status": 429,
                "outcome": "rejected",
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "status": 400,
                "outcome": "http_error",
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "status": None,
                "headers_ns": None,
                "outcome": "cancelled",
                "tokens": [],
                "token_received_ns": [],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "outcome": "cancelled",
                "tokens": [3],
                "token_received_ns": [2_000_000],
                "model_sha256": None,
                "finish_reason": None,
                "error_kind": "client_closed_stream",
            },
            {},
        ),
        ({"tokens": [], "token_received_ns": []}, {"cancellation": True}),
        (
            {"tokens": [3, 4], "token_received_ns": [2_000_000, 3_000_000]},
            {"cancellation": True},
        ),
        ({"error_kind": "ValueError"}, {}),
        ({"outcome": "protocol_error", "error_kind": "StreamProtocolError"}, {}),
        (
            {
                "outcome": "transport_error",
                "tokens": [],
                "token_received_ns": [],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        ({"outcome": "deadline"}, {}),
        ({"finished_ns": 21_000_000}, {}),
        (
            {
                "outcome": "deadline",
                "error_kind": "TimeoutError",
                "finished_ns": 21_000_000,
            },
            {},
        ),
        (
            {
                "outcome": "protocol_error",
                "error_kind": "StreamProtocolError",
                "tokens": [3],
                "model_sha256": None,
                "finish_reason": None,
            },
            {"streaming": False},
        ),
    ],
)
def test_resealed_reports_reject_states_the_collector_cannot_emit(changes, controls):
    with pytest.raises(ValueError):
        verify_http_load_report(forged_report(changes, **controls))


@pytest.mark.parametrize(
    "changes,controls",
    [
        ({}, {}),
        ({}, {"streaming": False}),
        ({}, {"cancellation": True}),
        (
            {
                "outcome": "cancelled",
                "finish_reason": "cancelled",
                "tokens": [],
                "token_received_ns": [],
            },
            {},
        ),
        (
            {
                "outcome": "deadline",
                "finish_reason": "deadline",
                "tokens": [3],
                "token_received_ns": [2_000_000],
            },
            {},
        ),
        (
            {
                "outcome": "deadline",
                "error_kind": "client_request_deadline",
                "finished_ns": 21_000_000,
            },
            {},
        ),
        (
            {
                "outcome": "deadline",
                "error_kind": "client_request_deadline",
                "finished_ns": 21_000_000,
                "status": None,
                "headers_ns": None,
                "tokens": [],
                "token_received_ns": [],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "outcome": "protocol_error",
                "error_kind": "StreamProtocolError",
                "tokens": [3],
                "token_received_ns": [2_000_000],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "outcome": "transport_error",
                "error_kind": "ConnectionResetError",
                "tokens": [3],
                "token_received_ns": [2_000_000],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "outcome": "rejected",
                "status": 429,
                "tokens": [],
                "token_received_ns": [],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
        (
            {
                "outcome": "completed",
                "finish_reason": "eos",
                "tokens": [3],
                "token_received_ns": [2_000_000],
                "model_sha256": "b" * 64,
                "error_kind": None,
            },
            {"cancellation": True, "cancel_after_tokens": 2},
        ),
        (
            {
                "outcome": "deadline",
                "error_kind": "client_request_deadline",
                "finished_ns": 21_000_000,
                "status": 429,
                "tokens": [],
                "token_received_ns": [],
                "model_sha256": None,
                "finish_reason": None,
            },
            {},
        ),
    ],
)
def test_contract_retains_valid_terminal_partial_and_deadline_override_records(
    changes, controls
):
    verify_http_load_report(forged_report(changes, **controls))
