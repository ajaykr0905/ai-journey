import copy

import pytest

from ai_journey.http_load import http_load_totals, run_http_load
from ai_journey.http_load_profile import HTTPWorkloadProfile
from ai_journey.http_load_report import (
    build_http_load_report,
    http_report_digest,
    verify_http_load_report,
)
from test_http_measurements import running_service


@pytest.fixture(scope="module")
def measured_report():
    profile = HTTPWorkloadProfile(
        measured_requests=2, warmup_requests=0, concurrency=1, max_new_tokens=2
    )
    with running_service() as (endpoint, service):
        report = build_http_load_report(run_http_load(endpoint, profile))
    assert verify_http_load_report(report) == report
    return report


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("measurement", "totals", "attempted_requests"), 2.0),
        (("measurement", "totals", "outcomes", "cancelled"), False),
        (("latencies", "completed_ttft_ns", "count"), 2.0),
        (("latencies", "completed_inter_token_gap_ns", "count"), 2.0),
    ],
)
def test_resealed_derived_evidence_rejects_numeric_type_drift(
    measured_report, path, replacement
):
    report = copy.deepcopy(measured_report)
    container = report
    for name in path[:-1]:
        container = container[name]
    container[path[-1]] = replacement
    report["report_sha256"] = http_report_digest(report)
    with pytest.raises(ValueError):
        verify_http_load_report(report)


@pytest.mark.parametrize("outcome", ["deadline", "cancelled"])
def test_partial_terminal_model_identity_cannot_differ_from_completed_work(
    measured_report, outcome
):
    raw = copy.deepcopy(measured_report["measurement"])
    observation = raw["measured"][1]["observation"]
    observation.update(
        outcome=outcome,
        finish_reason=outcome,
        tokens=observation["tokens"][:1],
        token_received_ns=observation["token_received_ns"][:1],
    )
    raw["totals"] = http_load_totals(
        raw["measured"],
        work_started_ns=raw["work_started_ns"],
        work_finished_ns=raw["work_finished_ns"],
    )
    assert verify_http_load_report(build_http_load_report(raw))
    observation["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model identities"):
        build_http_load_report(raw)
