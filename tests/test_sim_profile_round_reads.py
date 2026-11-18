"""Round 1: a buffer read is recorded inside DeviceBuffer.__getitem__."""

import pytest

from ai_journey.kernel_launch_buffer import AccessRecorder, DeviceBuffer
from ai_journey.kernel_launch_config import Dim3, KernelLaunchError, LaunchConfig
from ai_journey.sim_profile_harness import profile_call
from test_sim_profile_fixtures import small_config
from ai_journey.sim_profile_workloads import build_workloads

BUFFER = "ai_journey.kernel_launch_buffer"
CONFIG = LaunchConfig(Dim3(2), Dim3(4))


def test_a_read_calls_no_recorder_helper():
    matmul = build_workloads(small_config())[2]
    records, run = profile_call(matmul.checked_run)
    calls = {record.function: record.calls for record in records}
    log = run.result.log

    assert calls[f"{BUFFER}:DeviceBuffer.__getitem__"] == log.reads == 2 * 9 * 10 * 4
    assert f"{BUFFER}:AccessRecorder._record_read" not in calls
    # Only the writes still go through the helpers.
    assert calls[f"{BUFFER}:DeviceBuffer._recorder_for"] == log.writes == 90
    assert calls[f"{BUFFER}:AccessRecorder._running"] == log.writes


def test_read_checks_keep_their_order():
    data = DeviceBuffer("data", "uint8", [4, 5])

    with pytest.raises(KernelLaunchError, match="outside a kernel launch"):
        data[True]
    recorder = AccessRecorder(CONFIG, [data])
    # The index type is checked before the running thread, as before.
    with pytest.raises(TypeError, match="index must be an integer"):
        data[1.0]
    with pytest.raises(KernelLaunchError, match="only by a running thread"):
        data[9]
    recorder.enter(6)
    assert (data[1], data[2], data[-3]) == (5, 0, 0)
    recorder.leave()
    log = recorder.finish()

    assert log.reads == 1
    assert [(r.access, r.index, r.thread.x) for r in log.out_of_bounds] == [
        ("read", 2, 2),
        ("read", -3, 2),
    ]
