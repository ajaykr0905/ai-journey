import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from ai_journey.recovery_detection import (
    wait_for_process_exit,
    wait_for_progress_state,
)
from ai_journey.recovery_progress import WorkerProgress, write_worker_progress

# A worker's last act, publishing its final progress and exiting 0, held back
# until the test closes the worker's stdin.
PUBLISH_WHEN_RELEASED = """
import json
import sys
from pathlib import Path

from ai_journey.recovery_progress import WorkerProgress, write_worker_progress

sys.stdin.read()
write_worker_progress(Path(sys.argv[1]), WorkerProgress(**json.loads(sys.argv[2])))
"""


class Process:
    def __init__(self, results):
        self.results = iter(results)

    def poll(self):
        return next(self.results)


class StalledBeforePoll:
    """Polls a real worker like a supervisor descheduled after a progress read.

    The stall releases the worker and lasts until it has published its final
    progress and exited, so the first ``poll()`` reports an exit that came
    after the worker's last write.
    """

    def __init__(self, worker):
        self.worker = worker

    def poll(self):
        if self.worker.returncode is None:
            self.worker.stdin.close()
            self.worker.wait(timeout=30)
        return self.worker.poll()


class PublishesThenExits:
    """A worker that publishes ``last`` and exits between a read and ``poll()``."""

    def __init__(self, path, last, exit_code):
        self.path = path
        self.last = last
        self.exit_code = exit_code

    def poll(self):
        write_worker_progress(self.path, self.last)
        return self.exit_code


def progress():
    return WorkerProgress(
        run_id="day-70-worker-0123456789abcdef",
        attempt_id="attempt-1-0123456789abcdef",
        completed_items=11,
        accumulator_sha256="a" * 64,
        monotonic_ns=100,
        state="fault_ready",
    )


def test_wait_for_progress_returns_exact_observation(tmp_path):
    path = tmp_path / "progress.json"
    write_worker_progress(path, progress())
    ticks = iter([0, 1])
    assert (
        wait_for_progress_state(
            Process([None]),
            path,
            "fault_ready",
            timeout_ms=10,
            clock_ns=lambda: next(ticks),
            sleep=lambda _: None,
        )
        == progress()
    )


def test_wait_for_progress_fails_fast_on_worker_exit(tmp_path):
    with pytest.raises(ValueError, match="code 7"):
        wait_for_progress_state(
            Process([7]),
            tmp_path / "missing.json",
            "fault_ready",
            timeout_ms=10,
            clock_ns=lambda: 0,
            sleep=lambda _: None,
        )


def test_wait_for_progress_accepts_completion_published_during_a_poll_stall(
    tmp_path,
):
    # The real worker publishes "completed" and exits 0 while the supervisor
    # is stalled between reading the older progress and polling the process.
    path = tmp_path / "progress.json"
    write_worker_progress(path, progress())
    final = replace(progress(), completed_items=20, state="completed")
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PUBLISH_WHEN_RELEASED,
            str(path),
            json.dumps(asdict(final)),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        observed = wait_for_progress_state(
            StalledBeforePoll(worker), path, "completed", timeout_ms=30_000
        )
    finally:
        worker.kill()
        worker.wait()
        worker.stdin.close()

    assert observed == final
    assert worker.returncode == 0


@pytest.mark.parametrize(
    ("state", "exit_code"), [("running", 0), ("running", -9), ("fault_ready", 0)]
)
def test_wait_for_progress_still_fails_on_exit_short_of_the_state(
    tmp_path, state, exit_code
):
    path = tmp_path / "progress.json"
    write_worker_progress(path, replace(progress(), state="running"))
    last = replace(progress(), completed_items=19, state=state)
    with pytest.raises(ValueError, match=f"code {exit_code} before completed"):
        wait_for_progress_state(
            PublishesThenExits(path, last, exit_code),
            path,
            "completed",
            timeout_ms=10,
            clock_ns=lambda: 0,
            sleep=lambda _: None,
        )


def test_wait_for_progress_times_out_while_the_worker_runs(tmp_path):
    path = tmp_path / "progress.json"
    write_worker_progress(path, progress())
    ticks = iter([0, 0, 10_000_001])
    with pytest.raises(ValueError, match="did not reach completed before timeout"):
        wait_for_progress_state(
            Process([None]),
            path,
            "completed",
            timeout_ms=10,
            clock_ns=lambda: next(ticks),
            sleep=lambda _: None,
        )


def test_wait_for_process_exit_returns_detection_timestamp():
    ticks = iter([0, 1, 2, 3, 4])
    code, detected = wait_for_process_exit(
        Process([None, -9]),
        timeout_ms=10,
        clock_ns=lambda: next(ticks),
        sleep=lambda _: None,
    )
    assert code == -9
    assert detected == 3


def test_wait_for_progress_rejects_invalid_stored_evidence(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"state": "fault_ready"}))
    with pytest.raises(ValueError, match="fields"):
        wait_for_progress_state(
            Process([None]),
            path,
            "fault_ready",
            timeout_ms=10,
            clock_ns=lambda: 0,
            sleep=lambda _: None,
        )


@pytest.mark.parametrize("timeout", [True, 0])
def test_detection_rejects_invalid_timeouts(timeout):
    with pytest.raises((TypeError, ValueError), match="integer|positive"):
        wait_for_process_exit(Process([None]), timeout_ms=timeout)
