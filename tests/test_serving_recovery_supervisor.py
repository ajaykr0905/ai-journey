import os
import signal
import socket
import sys
from pathlib import Path

import pytest
from test_serving_recovery_config import valid_config

from ai_journey.serving_recovery_model import build_serving_model
from ai_journey.serving_recovery_server import server_command
from ai_journey.serving_recovery_supervisor import ServingSupervisor, probe_health

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "day-19-demo-names.txt"
MODEL = build_serving_model(FIXTURE.read_bytes())
RUN = "serving-recovery-test-0123456789abcdef"
# A long restart backoff that closing interrupts keeps liveness checks isolated.
NO_RESTART = {"restart_backoff_ms": 5000, "retry_budget": 22}


def real_server(tmp_path):
    def command(number, lifetime_id):
        return server_command(
            source_path=FIXTURE,
            journal_path=tmp_path / f"journal-{number}.jsonl",
            lifetime_id=lifetime_id,
            model_sha256=MODEL.model_sha256,
            max_new_tokens=8,
            max_records=64,
        )

    return command


def python(code, *arguments):
    return lambda number, lifetime_id: (
        sys.executable,
        "-I",
        "-c",
        code,
        lifetime_id,
        MODEL.model_sha256,
        *arguments,
    )


def supervisor(tmp_path, command=None, **changes):
    controls = {"probe_interval_ms": 10, "probe_timeout_ms": 100}
    return ServingSupervisor(
        run_id=RUN,
        config=valid_config(**(controls | {"detection_timeout_ms": 400} | changes)),
        model_sha256=MODEL.model_sha256,
        command_for=command or real_server(tmp_path),
        log_dir=tmp_path,
    )


def kinds(owner):
    return [observation.kind for observation in owner.observations()]


def observed(owner, kind):
    return any(observation.kind == kind for observation in owner.observations())


def exited(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_supervisor_launches_a_ready_lifetime_and_stops_it_cleanly(tmp_path):
    with supervisor(tmp_path) as owner:
        view = owner.start()
        assert (view.number, view.ready, view.exit_code) == (1, True, None)
        assert owner.endpoint() == f"http://127.0.0.1:{view.port}"
        assert probe_health(
            view.port,
            lifetime_id=view.lifetime_id,
            pid=view.pid,
            model_sha256=MODEL.model_sha256,
            timeout_ms=1000,
        )
        assert owner.stop() == 0

    assert kinds(owner) == ["spawn", "ready", "stopped"]
    times = [observation.monotonic_ns for observation in owner.observations()]
    assert times == sorted(times)
    assert (tmp_path / "lifetime-1.stderr").read_bytes() == b""
    assert exited(view.pid)


def test_exit_status_liveness_observes_a_killed_child(tmp_path):
    with supervisor(tmp_path, **NO_RESTART) as owner:
        view = owner.start()
        target, signaled_ns = owner.signal_current(view.lifetime_id, signal.SIGKILL)
        assert target == view
        assert owner.wait_until(
            lambda: observed(owner, "exit_observed"), timeout_ms=5000
        )
        with pytest.raises(ValueError, match="not ready"):
            owner.signal_current(view.lifetime_id, signal.SIGKILL)

    exit_observed = owner.observations()[2]
    assert kinds(owner)[:3] == ["spawn", "ready", "exit_observed"]
    assert exit_observed.exit_code == -signal.SIGKILL
    assert exit_observed.monotonic_ns >= signaled_ns


def test_health_probes_kill_a_live_but_unresponsive_child(tmp_path):
    with supervisor(tmp_path, **NO_RESTART) as owner:
        view = owner.start()
        _, stopped_ns = owner.signal_current(view.lifetime_id, signal.SIGSTOP)
        assert owner.wait_until(
            lambda: observed(owner, "exit_observed"), timeout_ms=10_000
        )

    assert kinds(owner)[:4] == ["spawn", "ready", "unresponsive", "exit_observed"]
    unresponsive, exit_observed = owner.observations()[2:4]
    assert unresponsive.monotonic_ns > stopped_ns
    assert exit_observed.exit_code == -signal.SIGKILL


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.parametrize(
    ("code", "changes", "message"),
    [
        ("raise SystemExit(3)", {}, "code 3"),
        ("import time; time.sleep(60)", {"readiness_timeout_ms": 200}, "in time"),
        (
            "import json, sys, time; print(json.dumps({'event': 'ready', "
            "'lifetime_id': sys.argv[1], 'model_sha256': sys.argv[2], 'pid': 1, "
            "'port': 80}, sort_keys=True, separators=(',', ':')), flush=True); "
            "time.sleep(60)",
            {},
            "spawned child",
        ),
        (
            "import json, os, sys, time; print(json.dumps({'event': 'ready', "
            "'lifetime_id': sys.argv[1], 'model_sha256': sys.argv[2], "
            "'pid': os.getpid(), 'port': int(sys.argv[3])}, sort_keys=True, "
            "separators=(',', ':')), flush=True); time.sleep(60)",
            {"readiness_timeout_ms": 300},
            "not healthy",
        ),
    ],
)
def test_readiness_failures_are_bounded_and_leave_no_child(
    tmp_path, code, changes, message
):
    owner = supervisor(tmp_path, python(code, str(closed_port())), **changes)
    with owner, pytest.raises(ValueError, match=message):
        owner.start()

    assert kinds(owner) == ["spawn", "exit_observed"]
    assert exited(owner.current().pid)


def test_signals_only_reach_the_current_ready_lifetime(tmp_path):
    with supervisor(tmp_path) as owner:
        with pytest.raises(ValueError, match="current lifetime"):
            owner.signal_current("lifetime-1-0123456789abcdef", signal.SIGKILL)
        view = owner.start()
        with pytest.raises(ValueError, match="current lifetime"):
            owner.signal_current("lifetime-2-0123456789abcdef", signal.SIGKILL)
        with pytest.raises(TypeError, match="signal"):
            owner.signal_current(view.lifetime_id, True)
        assert owner.current().exit_code is None


@pytest.mark.parametrize("argv", [(), (1,)])
def test_command_factories_must_return_argument_vectors(tmp_path, argv):
    owner = supervisor(tmp_path, lambda number, lifetime_id: argv)
    with owner, pytest.raises(TypeError, match="string sequence"):
        owner.start()
    with pytest.raises(ValueError, match="not started"):
        owner.current()
