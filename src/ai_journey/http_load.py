"""Bounded concurrent load against the actual loopback HTTP inference service."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import time
from uuid import uuid4

from .http_load_profile import HTTPWorkloadProfile, http_request_schedule
from .http_measurements import loopback_address, measure_http_generation

OUTCOMES = (
    "completed",
    "cancelled",
    "deadline",
    "rejected",
    "http_error",
    "protocol_error",
    "transport_error",
)
HTTP_LOAD_SCOPE = (
    "client-observed loopback HTTP on the bounded CPU lab; no production or GPU claim"
)


def http_load_totals(observations, *, work_started_ns, work_finished_ns):
    elapsed = work_finished_ns - work_started_ns
    if elapsed <= 0:
        raise ValueError("HTTP load work window must be positive")
    counts = Counter(row["observation"]["outcome"] for row in observations)
    if not set(counts) <= set(OUTCOMES):
        raise ValueError("unsupported HTTP request outcome")
    completed_tokens = sum(
        len(row["observation"]["tokens"])
        for row in observations
        if row["observation"]["outcome"] == "completed"
    )
    partial_tokens = sum(
        len(row["observation"]["tokens"])
        for row in observations
        if row["observation"]["outcome"] != "completed"
    )
    return {
        "attempted_requests": len(observations),
        "outcomes": {name: counts[name] for name in OUTCOMES},
        "completed_output_tokens": completed_tokens,
        "partial_observed_tokens": partial_tokens,
        "work_window_ns": elapsed,
        "completed_requests_per_second": counts["completed"] * 1e9 / elapsed,
        "completed_tokens_per_second": completed_tokens * 1e9 / elapsed,
        "attempted_requests_per_second": len(observations) * 1e9 / elapsed,
    }


def run_http_load(endpoint, profile: HTTPWorkloadProfile):
    """Use fresh IDs and at most concurrency outstanding measured requests.

    Warmup attempts are retained separately. The measured denominator includes
    client dispatch, connection setup, queueing, failures and executor cleanup.
    Client cancellation means receipt stopped; it does not prove server work
    stopped. Requests are never retried or silently removed from the report.
    """

    loopback_address(endpoint)
    if not isinstance(profile, HTTPWorkloadProfile):
        raise TypeError("HTTP load requires a validated workload profile")
    run_id = uuid4().hex
    created = datetime.now(timezone.utc).isoformat()

    def execute(row):
        observation = measure_http_generation(
            endpoint,
            row["request"],
            streaming=profile.streaming,
            timeout_seconds=profile.timeout_seconds,
            cancel_after_tokens=row["cancel_after_tokens"],
        )
        return {"index": row["index"], "observation": observation}

    warmup = [
        execute(row) for row in http_request_schedule(profile, run_id, phase="warmup")
    ]
    schedule = iter(http_request_schedule(profile, run_id))
    measured = []
    work_started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=profile.concurrency) as pool:
        pending = {
            pool.submit(execute, next(schedule)) for _ in range(profile.concurrency)
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                measured.append(future.result())
                row = next(schedule, None)
                if row is not None:
                    pending.add(pool.submit(execute, row))
    work_finished = time.perf_counter_ns()
    measured.sort(key=lambda row: row["index"])
    return {
        "schema": "ai-journey-http-load-raw-v1",
        "scope": HTTP_LOAD_SCOPE,
        "created_at_utc": created,
        "endpoint": endpoint,
        "run_id": run_id,
        "profile": profile.to_mapping(),
        "profile_sha256": profile.fingerprint(),
        "warmup": warmup,
        "measured": measured,
        "work_started_ns": work_started,
        "work_finished_ns": work_finished,
        "totals": http_load_totals(
            measured, work_started_ns=work_started, work_finished_ns=work_finished
        ),
    }
