import pytest
from test_recovery_config import valid_config

from ai_journey.recovery_fault import inject_process_kill
from ai_journey.recovery_progress import WorkerProgress


class Process:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.killed = False

    def poll(self):
        return self.exit_code

    def kill(self):
        self.killed = True


def progress(**changes):
    values = {
        "run_id": "day-70-worker-0123456789abcdef",
        "attempt_id": "attempt-1-0123456789abcdef",
        "completed_items": valid_config().fault_after_items,
        "accumulator_sha256": "a" * 64,
        "monotonic_ns": 100,
        "state": "fault_ready",
    }
    return WorkerProgress(**(values | changes))


def test_fault_injection_kills_live_worker_at_declared_boundary():
    process = Process()

    fault = inject_process_kill(
        process, progress(), valid_config(), clock_ns=lambda: 123
    )

    assert process.killed is True
    assert fault.attempt_id == progress().attempt_id
    assert fault.completed_items == valid_config().fault_after_items
    assert fault.injected_monotonic_ns == 123
    assert fault.mechanism == "process_kill"


@pytest.mark.parametrize(
    ("worker", "observation", "message"),
    [
        (Process(), progress(state="running"), "fault_ready"),
        (Process(), progress(completed_items=10), "boundary"),
        (Process(exit_code=0), progress(), "exited"),
    ],
)
def test_fault_injection_rejects_unobserved_or_late_actions(
    worker, observation, message
):
    with pytest.raises(ValueError, match=message):
        inject_process_kill(worker, observation, valid_config())
    assert worker.killed is False
