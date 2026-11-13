import signal
import sys

from test_serving_recovery_supervisor import (
    MODEL,
    exited,
    kinds,
    real_server,
    supervisor,
)

from ai_journey.serving_recovery_journal import read_execution_journal
from ai_journey.serving_recovery_supervisor import probe_health


def failing_after_first(tmp_path, *, healthy=()):
    def command(number, lifetime_id):
        if number == 1 or number in healthy:
            return real_server(tmp_path)(number, lifetime_id)
        return (sys.executable, "-I", "-c", "raise SystemExit(4)")

    return command


def kill_current(owner):
    view = owner.current()
    owner.signal_current(view.lifetime_id, signal.SIGKILL)
    return view


def test_killed_lifetime_is_replaced_after_its_declared_backoff(tmp_path):
    with supervisor(tmp_path) as owner:
        first = owner.start()
        kill_current(owner)
        assert owner.wait_until(
            lambda: owner.current().number == 2 and owner.current().ready,
            timeout_ms=10_000,
        )
        second = owner.current()
        assert owner.endpoint() == second.endpoint
        assert probe_health(
            second.port,
            lifetime_id=second.lifetime_id,
            pid=second.pid,
            model_sha256=MODEL.model_sha256,
            timeout_ms=1000,
        )
        assert owner.stop() == 0

    assert kinds(owner) == [
        "spawn",
        "ready",
        "exit_observed",
        "restart",
        "ready",
        "stopped",
    ]
    exit_observed, restart = owner.observations()[2:4]
    assert restart.backoff_ms == owner.config.restart_delay_ms(1)
    assert restart.monotonic_ns >= exit_observed.monotonic_ns
    assert (second.pid, second.lifetime_id) != (first.pid, first.lifetime_id)
    assert owner.restarts_used == 1 and owner.failure is None
    for number in (1, 2):
        read_execution_journal(
            tmp_path / f"journal-{number}.jsonl",
            lifetime_id=owner.observations()[0 if number == 1 else 3].lifetime_id,
            max_records=64,
            allow_torn_tail=number == 1,
        )


def test_failed_replacements_consume_the_bounded_restart_budget(tmp_path):
    with supervisor(tmp_path, failing_after_first(tmp_path)) as owner:
        owner.start()
        kill_current(owner)
        assert owner.wait_until(lambda: owner.failure is not None, timeout_ms=10_000)

    assert kinds(owner) == [
        "spawn",
        "ready",
        "exit_observed",
        "restart",
        "exit_observed",
        "restart",
        "exit_observed",
        "exhausted",
    ]
    restarts = [o for o in owner.observations() if o.kind == "restart"]
    assert [o.backoff_ms for o in restarts] == [20, 40]
    assert [o.lifetime_number for o in restarts] == [2, 3]
    assert [o.exit_code for o in owner.observations() if o.kind == "exit_observed"][
        1:
    ] == [4, 4]
    assert "exhausted" in str(owner.failure)
    assert all(exited(o.pid) for o in owner.observations())


def test_a_failed_replacement_is_followed_by_a_ready_one(tmp_path):
    command = failing_after_first(tmp_path, healthy={3})
    with supervisor(tmp_path, command) as owner:
        owner.start()
        kill_current(owner)
        assert owner.wait_until(
            lambda: owner.current().number == 3 and owner.current().ready,
            timeout_ms=10_000,
        )
        assert owner.stop() == 0

    assert kinds(owner)[2:] == [
        "exit_observed",
        "restart",
        "exit_observed",
        "restart",
        "ready",
        "stopped",
    ]
    assert owner.restarts_used == 2 and owner.failure is None


def test_stop_interrupts_a_pending_restart_backoff(tmp_path):
    owner = supervisor(
        tmp_path, restart_backoff_ms=5000, retry_budget=23, restart_limit=1
    )
    with owner:
        view = owner.start()
        kill_current(owner)
        assert owner.wait_until(
            lambda: owner.current().exit_code is not None, timeout_ms=5000
        )
        assert owner.stop() == -signal.SIGKILL

    assert kinds(owner) == ["spawn", "ready", "exit_observed"]
    assert owner.current() == view.__class__(
        1, view.lifetime_id, view.pid, view.port, False, -signal.SIGKILL
    )
