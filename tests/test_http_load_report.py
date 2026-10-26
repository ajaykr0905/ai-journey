import copy

import pytest

from ai_journey.ddp_service import TokenBucket
from ai_journey.http_load import run_http_load
from ai_journey.http_load_profile import HTTPWorkloadProfile
from ai_journey.http_load_report import (
    build_http_load_report,
    http_report_digest,
    verify_http_load_report,
)
from test_http_measurements import running_service


def report():
    profile = HTTPWorkloadProfile(
        measured_requests=3, warmup_requests=1, concurrency=2, max_new_tokens=3
    )
    with running_service() as (endpoint, service):
        return build_http_load_report(run_http_load(endpoint, profile))


def test_sealed_report_recomputes_token_timing_distributions_and_real_denominator():
    evidence = report()
    assert verify_http_load_report(evidence) == evidence
    assert evidence["latencies"]["completed_ttft_ns"]["count"] == 3
    assert evidence["latencies"]["completed_inter_token_gap_ns"]["count"] == 6
    assert (
        evidence["latencies"]["all_request_duration_ns"]["p95"]
        == evidence["latencies"]["all_request_duration_ns"]["maximum"]
    )


def test_even_resealed_tampering_cannot_change_derived_results_or_work_coverage():
    evidence = report()
    for mutate in (
        lambda changed: changed["measurement"]["totals"].update(
            completed_output_tokens=0
        ),
        lambda changed: changed["latencies"]["completed_ttft_ns"].update(p95=-1),
        lambda changed: changed["measurement"]["measured"][0]["observation"][
            "token_received_ns"
        ].__setitem__(0, 0),
        lambda changed: changed["measurement"]["measured"].pop(),
    ):
        changed = copy.deepcopy(evidence)
        mutate(changed)
        changed["report_sha256"] = http_report_digest(changed)
        with pytest.raises(ValueError):
            verify_http_load_report(changed)


def test_all_rejected_work_keeps_empty_success_distributions_and_attempt_denominator():
    profile = HTTPWorkloadProfile(
        measured_requests=2, warmup_requests=1, concurrency=1, max_new_tokens=3
    )
    with running_service(admission=TokenBucket(5, 0.00001)) as (endpoint, service):
        evidence = build_http_load_report(run_http_load(endpoint, profile))
    verify_http_load_report(evidence)
    assert evidence["measurement"]["totals"]["outcomes"]["rejected"] == 2
    assert evidence["measurement"]["totals"]["completed_tokens_per_second"] == 0
    assert evidence["latencies"]["completed_ttft_ns"]["p95"] is None
