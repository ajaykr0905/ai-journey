import json

import pytest

from ai_journey.recovery_detection import (
    wait_for_process_exit,
    wait_for_progress_state,
)
from ai_journey.recovery_progress import WorkerProgress, write_worker_progress


class Process:
    def __init__(self, results):
        self.results = iter(results)

    def poll(self):
        return next(self.results)


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
