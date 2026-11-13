import itertools
import threading
import time

import pytest
from test_serving_recovery_client import (
    LIFETIME,
    MODEL,
    PLAN,
    closed_port,
    live_server,
)
from test_serving_recovery_config import valid_config

from ai_journey.serving_recovery_client import AttemptRecord
from ai_journey.serving_recovery_journal import read_execution_journal
from ai_journey.serving_recovery_load import AttemptLedger, run_load, run_with_retries

CONFIG = valid_config()
FAR = time.monotonic_ns() + 3600 * 10**9


def record(planned, attempt, outcome, *, port=8080, tick=itertools.count()):
    started = next(tick)
    ok = outcome == "ok"
    return AttemptRecord(
        request_index=planned.index,
        idempotency_key=planned.idempotency_key,
        attempt=attempt,
        port=port,
        started_ns=started,
        finished_ns=started + 1,
        outcome=outcome,
        error=None if ok else "scripted",
        http_status=200 if ok else None,
        lifetime_id=LIFETIME if ok else None,
        response_sha256="e" * 64 if ok else None,
    )


def scripted(outcomes, calls):
    def send(endpoint, planned, *, attempt, **options):
        calls.append((endpoint, planned.idempotency_key, attempt, options))
        return record(planned, attempt, outcomes[attempt - 1])

    return send


def retry(send, *, endpoint=lambda: "http://127.0.0.1:8080", waits=None, **changes):
    waits = [] if waits is None else waits
    return run_with_retries(
        PLAN[3],
        endpoint=endpoint,
        config=valid_config(**changes),
        model_sha256=MODEL.model_sha256,
        ledger=AttemptLedger(),
        deadline_ns=FAR,
        wait=lambda seconds: waits.append(seconds) or False,
        send=send,
    )


def test_retries_carry_one_key_and_follow_the_backoff_schedule():
    calls, waits, ledger = [], [], AttemptLedger()
    outcomes = ["connection_reset", "connection_refused", "connection_refused", "ok"]
    final = run_with_retries(
        PLAN[3],
        endpoint=lambda: f"http://127.0.0.1:{8080 + len(calls)}",
        config=CONFIG,
        model_sha256=MODEL.model_sha256,
        ledger=ledger,
        deadline_ns=FAR,
        wait=lambda seconds: waits.append(seconds) or False,
        send=scripted(outcomes, calls),
    )

    assert final.outcome == "ok" and final.attempt == 4
    assert {key for _, key, _, _ in calls} == {PLAN[3].idempotency_key}
    assert [attempt for _, _, attempt, _ in calls] == [1, 2, 3, 4]
    assert [endpoint for endpoint, _, _, _ in calls] == [
        f"http://127.0.0.1:{8080 + n}" for n in range(4)
    ]
    assert {options["deadline_ms"] for *_, options in calls} == {1000}
    assert waits == [0.01, 0.02, 0.04]
    assert [a.outcome for a in ledger.attempts()] == outcomes
    assert ledger.acknowledged == 1


def test_requests_fail_only_after_spending_the_whole_retry_budget():
    calls, waits = [], []
    final = retry(scripted(["timeout"] * 15, calls), waits=waits)

    assert (final.outcome, final.attempt) == ("timeout", 15)
    assert len(calls) == CONFIG.retry_budget + 1
    assert waits == [CONFIG.retry_delay_ms(n) / 1000 for n in range(1, 15)]


def test_retries_stop_at_the_wall_clock_bound_or_a_stop_request():
    with pytest.raises(ValueError, match="stopped before"):
        run_with_retries(
            PLAN[0],
            endpoint=lambda: "http://127.0.0.1:8080",
            config=CONFIG,
            model_sha256=MODEL.model_sha256,
            ledger=AttemptLedger(),
            deadline_ns=0,
            send=scripted(["ok"], []),
        )
    with pytest.raises(ValueError, match="backoff"):
        run_with_retries(
            PLAN[0],
            endpoint=lambda: "http://127.0.0.1:8080",
            config=CONFIG,
            model_sha256=MODEL.model_sha256,
            ledger=AttemptLedger(),
            deadline_ns=FAR,
            wait=lambda seconds: True,
            send=scripted(["timeout", "ok"], []),
        )


def test_ledger_hook_sees_each_acknowledged_count_exactly_once_in_order():
    seen = []
    ledger = AttemptLedger(
        on_acknowledged=lambda count: seen.append((count, len(ledger.attempts())))
    )

    def submit(base):
        for offset in range(25):
            planned = PLAN[(base * 25 + offset) % len(PLAN)]
            ledger.record(record(planned, 1, "ok" if offset % 2 else "timeout"))

    threads = [threading.Thread(target=submit, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert [count for count, _ in seen] == list(range(1, 49))
    attempts = ledger.attempts()
    for count, recorded in seen:
        assert sum(a.outcome == "ok" for a in attempts[:recorded]) == count


def test_load_bounds_concurrency_and_finishes_every_request():
    active, peak, lock = [0], [0], threading.Lock()

    def send(endpoint, planned, *, attempt, **options):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.002)
        with lock:
            active[0] -= 1
        return record(planned, attempt, "ok" if attempt == 2 else "connection_reset")

    ledger = AttemptLedger()
    attempts = run_load(
        PLAN,
        endpoint=lambda: "http://127.0.0.1:8080",
        config=valid_config(retry_backoff_ms=1, retry_budget=20),
        model_sha256=MODEL.model_sha256,
        ledger=ledger,
        deadline_ns=FAR,
        send=send,
    )

    assert peak[0] <= CONFIG.concurrency
    assert len(attempts) == 2 * len(PLAN)
    assert sorted(a.request_index for a in attempts if a.outcome == "ok") == list(
        range(len(PLAN))
    )


def test_load_surfaces_sender_failures_and_rejects_partial_plans():
    def broken(endpoint, planned, *, attempt, **options):
        raise RuntimeError("sender failed")

    options = {
        "endpoint": lambda: "http://127.0.0.1:8080",
        "config": CONFIG,
        "model_sha256": MODEL.model_sha256,
        "ledger": AttemptLedger(),
        "deadline_ns": FAR,
    }
    with pytest.raises(RuntimeError, match="sender failed"):
        run_load(PLAN, send=broken, **options)
    with pytest.raises(ValueError, match="every request"):
        run_load(PLAN[1:], send=broken, **options)
    with pytest.raises(ValueError, match="already passed"):
        run_load(PLAN, send=broken, **(options | {"deadline_ns": 0}))


def test_a_refused_request_retries_onto_a_live_server_with_the_same_key(tmp_path):
    ports = [closed_port()]
    ledger = AttemptLedger()
    with live_server(tmp_path) as port:
        final = run_with_retries(
            PLAN[2],
            endpoint=lambda: f"http://127.0.0.1:{ports[-1]}",
            config=CONFIG,
            model_sha256=MODEL.model_sha256,
            ledger=ledger,
            deadline_ns=FAR,
            wait=lambda seconds: ports.append(port) and False,
        )

    first, second = ledger.attempts()
    assert (first.outcome, second.outcome) == ("connection_refused", "ok")
    assert final == second and second.lifetime_id == LIFETIME
    (execution,) = read_execution_journal(
        tmp_path / "journal.jsonl",
        lifetime_id=LIFETIME,
        max_records=16,
        allow_torn_tail=False,
    ).records
    assert (execution.idempotency_key, execution.attempt) == (
        PLAN[2].idempotency_key,
        2,
    )
