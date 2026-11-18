"""Round 3: finish names threads only for elements that race."""

from ai_journey.kernel_launch_buffer import AccessRecorder, DeviceBuffer
from ai_journey.kernel_launch_config import Dim3, Index3, LaunchConfig
from ai_journey.kernel_launch_kernels import run_vector_add
from ai_journey.kernel_launch_reference import vector_add_inputs
from ai_journey.sim_profile_harness import profile_call

INDEX3 = "ai_journey.kernel_launch_config:Index3.__post_init__"
CONFIG = LaunchConfig(Dim3(2), Dim3(4))


def test_a_race_free_launch_builds_no_thread_refs_in_finish():
    a, b = vector_add_inputs(75, 1000)
    records, run = profile_call(lambda: run_vector_add(a, b, threads_per_block=256))
    calls = {record.function: record.calls for record in records}

    assert run.result.log.race_free
    assert run.result.log.writer_threads["out"] == 1000
    assert "ai_journey.kernel_launch_buffer:AccessRecorder._ref" not in calls
    # Only the launch's own tables remain: coordinates() gives two Index3
    # values for each of 4 blocks and 256 threads. Before round 3, finish
    # added two more for each of the 1000 written elements.
    assert calls[INDEX3] == 2 * (4 + 256)


def test_every_kind_of_element_keeps_its_record():
    out = DeviceBuffer("out", "uint8", [0] * 6)
    recorder = AccessRecorder(CONFIG, [out])
    # Elements 0, 1 and 4 are race-free: one writer, at most its own reads.
    plan = {
        0: [("w", 0)],
        1: [("w", 1), ("w", 1)],
        2: [("w", 2), ("r", 2)],
        3: [("r", 2), ("w", 3)],
        4: [("w", 4), ("r", 3)],
        5: [("w", 5)],
        6: [("w", 5)],
    }
    for thread_id, steps in plan.items():
        recorder.enter(thread_id)
        for kind, index in steps:
            if kind == "w":
                out[index] = thread_id
            else:
                out[index]
        recorder.leave()
    log = recorder.finish()

    assert [
        (item.index, [writer.thread for writer in item.writers])
        for item in log.write_conflicts
    ] == [(5, [Index3(1), Index3(2)])]
    assert [
        (
            item.index,
            [writer.thread for writer in item.writers],
            [reader.thread for reader in item.readers],
        )
        for item in log.read_write_conflicts
    ] == [
        (2, [Index3(2)], [Index3(2), Index3(3)]),
        (3, [Index3(3)], [Index3(0)]),
    ]
    assert log.write_conflicts[0].writers[0].block == Index3(1)
    assert log.writer_threads == {"out": 7}
    assert (log.reads, log.writes) == (3, 8)
