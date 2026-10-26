from ai_journey.ddp_service import TokenBucket
from ai_journey.http_load import run_http_load
from ai_journey.http_load_profile import HTTPWorkloadProfile
from test_http_measurements import running_service


def test_real_concurrent_load_uses_actual_work_window_and_fresh_ids():
    profile = HTTPWorkloadProfile(
        measured_requests=4, warmup_requests=1, concurrency=2, max_new_tokens=3
    )
    with running_service() as (endpoint, service):
        first = run_http_load(endpoint, profile)
        second = run_http_load(endpoint, profile)
    assert first["run_id"] != second["run_id"]
    assert service.completed_requests == 10
    assert len(first["warmup"]) == 1
    assert len(first["measured"]) == 4
    assert first["totals"]["outcomes"]["completed"] == 4
    assert first["totals"]["completed_output_tokens"] == 12
    duration = first["work_finished_ns"] - first["work_started_ns"]
    assert first["totals"]["completed_tokens_per_second"] == 12 * 1e9 / duration
    assert first["warmup"][-1]["observation"]["finished_ns"] <= first["work_started_ns"]
    for row in first["measured"]:
        assert (
            first["work_started_ns"]
            <= row["observation"]["started_ns"]
            <= row["observation"]["finished_ns"]
            <= first["work_finished_ns"]
        )


def test_rejections_remain_in_the_denominator_and_cancellation_is_counted():
    profile = HTTPWorkloadProfile(
        measured_requests=3, warmup_requests=0, concurrency=1, max_new_tokens=3
    )
    with running_service(admission=TokenBucket(5, 0.00001)) as (endpoint, service):
        report = run_http_load(endpoint, profile)
    assert report["totals"]["attempted_requests"] == 3
    assert report["totals"]["outcomes"]["completed"] == 1
    assert report["totals"]["outcomes"]["rejected"] == 2
    profile = HTTPWorkloadProfile(
        measured_requests=4,
        warmup_requests=0,
        concurrency=2,
        max_new_tokens=3,
        cancel_every=2,
    )
    with running_service() as (endpoint, service):
        report = run_http_load(endpoint, profile)
    assert report["totals"]["outcomes"]["cancelled"] == 2
    assert report["totals"]["completed_output_tokens"] == 6
    assert report["totals"]["partial_observed_tokens"] == 2
