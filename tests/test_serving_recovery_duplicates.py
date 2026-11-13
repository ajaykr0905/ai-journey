import os
import signal
import threading
import time

from test_serving_recovery_client import MODEL, PLAN, RUN
from test_serving_recovery_config import valid_config
from test_serving_recovery_supervisor import FIXTURE

from ai_journey.serving_recovery_client import send_generation_attempt
from ai_journey.serving_recovery_journal import read_execution_journal
from ai_journey.serving_recovery_load import AttemptLedger, run_with_retries
from ai_journey.serving_recovery_reconcile import reconcile_serving_run
from ai_journey.serving_recovery_server import server_command
from ai_journey.serving_recovery_supervisor import ServingSupervisor

# A long per-attempt deadline keeps the first attempt in flight until the kill.
CONFIG = valid_config(request_deadline_ms=5000, max_run_ms=120_000)


def commands(tmp_path, held_lifetimes):
    def command(number, lifetime_id):
        return server_command(
            source_path=FIXTURE,
            journal_path=tmp_path / f"journal-{number}.jsonl",
            lifetime_id=lifetime_id,
            model_sha256=MODEL.model_sha256,
            max_new_tokens=8,
            max_records=64,
            hold_response_ms=10_000 if number in held_lifetimes else 0,
        )

    return command


def wait_for_commit(path, timeout=10.0):
    """Poll, with a bound, until a journal holds one committed record."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_bytes().endswith(b"\n"):
            return
        time.sleep(0.005)
    raise AssertionError("the server never committed the held execution")


def journal(tmp_path, number, lifetime_id):
    return read_execution_journal(
        tmp_path / f"journal-{number}.jsonl",
        lifetime_id=lifetime_id,
        max_records=64,
        allow_torn_tail=number == 1,
    )


def test_a_kill_after_commit_loses_the_response_and_retries_duplicate_it(tmp_path):
    ledger = AttemptLedger()
    owner = ServingSupervisor(
        run_id=RUN,
        config=CONFIG,
        model_sha256=MODEL.model_sha256,
        command_for=commands(tmp_path, held_lifetimes={1}),
    )
    with owner:
        first = owner.start()
        worker = threading.Thread(
            target=run_with_retries,
            args=(PLAN[0],),
            kwargs={
                "endpoint": owner.endpoint,
                "config": CONFIG,
                "model_sha256": MODEL.model_sha256,
                "ledger": ledger,
                "deadline_ns": time.monotonic_ns() + 60 * 10**9,
            },
        )
        worker.start()
        wait_for_commit(tmp_path / "journal-1.jsonl")
        owner.signal_current(first.lifetime_id, signal.SIGKILL)
        worker.join(timeout=30)
        assert not worker.is_alive()
        second = owner.current()

    attempts = ledger.attempts()
    assert attempts[0].outcome in {"connection_reset", "timeout"}
    assert attempts[-1].outcome == "ok"
    assert attempts[-1].lifetime_id == second.lifetime_id != first.lifetime_id
    journals = [journal(tmp_path, 1, first.lifetime_id)]
    journals.append(journal(tmp_path, 2, second.lifetime_id))
    assert [len(item.records) for item in journals] == [1, 1]
    result = reconcile_serving_run(
        plan=PLAN[:1],
        attempts=attempts,
        journals=journals,
        retry_budget=CONFIG.retry_budget,
        model_sha256=MODEL.model_sha256,
    )
    (row,) = result.requests
    assert (row.outcome, row.executions, row.lost_response_executions) == (
        "success_after_retry",
        2,
        1,
    )
    assert result.totals["duplicate_executions"] == 1
    assert result.totals["requests_executed_more_than_once"] == 1
    executed = {entry.response_sha256 for item in journals for entry in item.records}
    assert executed == {attempts[-1].response_sha256}


def test_sigterm_releases_held_responses_before_draining(tmp_path):
    owner = ServingSupervisor(
        run_id=RUN,
        config=CONFIG,
        model_sha256=MODEL.model_sha256,
        command_for=commands(tmp_path, held_lifetimes={1}),
    )
    results = []
    with owner:
        view = owner.start()
        worker = threading.Thread(
            target=lambda: results.append(
                send_generation_attempt(
                    view.endpoint,
                    PLAN[1],
                    attempt=1,
                    deadline_ms=5000,
                    model_sha256=MODEL.model_sha256,
                )
            )
        )
        worker.start()
        wait_for_commit(tmp_path / "journal-1.jsonl")
        os.killpg(view.pid, signal.SIGTERM)
        worker.join(timeout=10)
        assert owner.wait_until(
            lambda: owner.observations()[-1].kind == "exit_observed", timeout_ms=5000
        )

    assert results[0].outcome == "ok"
    assert owner.observations()[2].exit_code == 0
