from concurrent.futures import ThreadPoolExecutor
import math
import re
import threading

import pytest

from ai_journey.serving_metrics import DURATION_BUCKETS, OUTCOMES, ServingMetrics


def test_serving_histogram_and_cache_counts_match_independent_oracle():
    metrics = ServingMetrics()
    observations = [
        ("success", 0.005, 4, False),
        ("cancelled", 0.1, 2, False),
        ("deadline", 3.0, 1, False),
        ("success", 0.0, 4, True),
    ]
    for outcome, duration, tokens, cached in observations:
        metrics.record(
            outcome=outcome,
            duration_seconds=duration,
            generated_tokens=tokens,
            cached=cached,
        )
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == dict(
        success=2, cancelled=1, deadline=1, error=0, rejected=0
    )
    assert snapshot["generated_tokens"] == 7
    assert snapshot["cache_hits"] == 1
    assert snapshot["duration_seconds"]["sum"] == pytest.approx(3.105)
    assert snapshot["duration_seconds"]["count"] == 4
    for bound, count in snapshot["duration_seconds"]["buckets"]:
        assert count == sum(duration <= bound for _, duration, _, _ in observations)


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "request-private-id"},
        {"outcome": ["success"]},
        {"duration_seconds": -1},
        {"duration_seconds": float("nan")},
        {"duration_seconds": float("inf")},
        {"duration_seconds": True},
        {"duration_seconds": "0.1"},
        {"duration_seconds": 10**400},
        {"generated_tokens": -1},
        {"generated_tokens": True},
        {"generated_tokens": 1.0},
        {"generated_tokens": 2**63},
        {"cached": 1},
    ],
)
def test_invalid_observation_does_not_partially_mutate_metrics(change):
    metrics = ServingMetrics()
    before = metrics.snapshot()
    with pytest.raises(ValueError):
        metrics.record(
            **{
                "outcome": "success",
                "duration_seconds": 0.1,
                "generated_tokens": 1,
                **change,
            }
        )
    assert metrics.snapshot() == before


def test_accumulation_overflow_is_atomic_and_snapshot_is_independent():
    metrics = ServingMetrics()
    metrics.record(
        outcome="success", duration_seconds=1e308, generated_tokens=2**63 - 1
    )
    before = metrics.snapshot()
    for duration, tokens in ((1e308, 0), (0, 1)):
        with pytest.raises(ValueError, match="finite capacity"):
            metrics.record(
                outcome="success", duration_seconds=duration, generated_tokens=tokens
            )
        assert metrics.snapshot() == before
    before["requests"]["success"] = 42
    assert metrics.snapshot()["requests"]["success"] == 1


def _samples(text):
    result = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            assert name not in result
            result[name] = float(value)
    return result


def test_prometheus_scrape_is_complete_bounded_and_cumulative():
    metrics = ServingMetrics()
    metrics.record(outcome="error", duration_seconds=0.1, generated_tokens=3)
    text = metrics.render_prometheus()
    assert text.endswith("\n")
    samples = _samples(text)
    assert len(samples) == len(OUTCOMES) + 2 + 2 * (len(DURATION_BUCKETS) + 3)
    assert samples['ai_journey_serving_requests_total{outcome="error"}'] == 1
    assert (
        samples['ai_journey_serving_request_duration_seconds_bucket{le="+Inf"}']
        == samples["ai_journey_serving_request_duration_seconds_count"]
        == 1
    )
    assert samples["ai_journey_serving_generated_tokens_total"] == 3
    assert all(math.isfinite(value) and value >= 0 for value in samples.values())
    assert set(re.findall(r"([a-z_]+)=", text)) == {"outcome", "le"}
    assert "request_id" not in text and "prompt" not in text
    for family in (
        "requests_total",
        "generated_tokens_total",
        "cache_hits_total",
        "request_duration_seconds",
        "queue_wait_seconds",
    ):
        prefix = f"ai_journey_serving_{family}"
        assert text.count(f"# TYPE {prefix} ") == 1
        assert text.index(f"# TYPE {prefix} ") < text.index(f"\n{prefix}")


def test_concurrent_writers_and_scrapers_observe_coherent_totals():
    metrics = ServingMetrics()
    barrier = threading.Barrier(9)

    def writer(index):
        barrier.wait()
        for iteration in range(500):
            metrics.record(
                outcome=OUTCOMES[index % len(OUTCOMES)],
                duration_seconds=0.05,
                generated_tokens=2,
                cached=iteration % 2 == 0,
            )
            metrics.observe_queue_wait(0.025)

    def scraper():
        barrier.wait()
        for _ in range(500):
            snapshot = metrics.snapshot()
            count = snapshot["duration_seconds"]["count"]
            assert sum(snapshot["requests"].values()) == count
            assert snapshot["duration_seconds"]["buckets"][-1][1] == count
            assert 0 <= snapshot["cache_hits"] <= count
            queue_wait = snapshot["queue_wait_seconds"]
            assert queue_wait["buckets"][-1][1] == queue_wait["count"]
            _samples(metrics.render_prometheus())

    with ThreadPoolExecutor(max_workers=9) as pool:
        tasks = [pool.submit(writer, index) for index in range(8)] + [
            pool.submit(scraper)
        ]
        for task in tasks:
            task.result(timeout=10)
    snapshot = metrics.snapshot()
    assert snapshot["duration_seconds"]["count"] == 4000
    assert snapshot["generated_tokens"] == 4000
    assert snapshot["cache_hits"] == 2000
    assert snapshot["queue_wait_seconds"]["count"] == 4000


def test_queue_wait_histogram_observes_waits_without_completed_requests():
    metrics = ServingMetrics()
    observations = (0.0, 0.025, 0.5, 60.0)
    for duration in observations:
        metrics.observe_queue_wait(duration)
    snapshot = metrics.snapshot()
    assert snapshot["duration_seconds"]["count"] == 0
    queue_wait = snapshot["queue_wait_seconds"]
    assert queue_wait["sum"] == sum(observations)
    assert queue_wait["count"] == len(observations)
    for bound, count in queue_wait["buckets"]:
        assert count == sum(duration <= bound for duration in observations)


@pytest.mark.parametrize(
    "duration", [-1, float("nan"), float("inf"), True, "1", 10**400]
)
def test_invalid_queue_wait_is_atomic(duration):
    metrics = ServingMetrics()
    before = metrics.snapshot()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        metrics.observe_queue_wait(duration)
    assert metrics.snapshot() == before


def test_queue_wait_overflow_is_atomic():
    metrics = ServingMetrics()
    metrics.observe_queue_wait(1e308)
    before = metrics.snapshot()
    with pytest.raises(ValueError, match="finite capacity"):
        metrics.observe_queue_wait(1e308)
    assert metrics.snapshot() == before


@pytest.mark.parametrize(
    "change",
    [
        {"active_requests": True},
        {"running_requests": -1},
        {"queued_requests": 1.0},
        {"request_capacity": 65},
        {"draining": 1},
        {"active_requests": 1},
        {"request_capacity": 0},
    ],
)
def test_operational_gauges_reject_incoherent_or_unbounded_counts(change):
    metrics = ServingMetrics()
    with pytest.raises(ValueError, match="operational state"):
        metrics.render_prometheus(
            operational_state={
                "active_requests": 0,
                "running_requests": 0,
                "queued_requests": 0,
                "request_capacity": 2,
                "draining": False,
                **change,
            }
        )
