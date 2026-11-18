import math
import struct
from hashlib import sha256

import pytest

from ai_journey.kernel_launch_buffer import (
    MAX_BUFFER_ELEMENTS,
    AccessRecorder,
    DeviceBuffer,
    OutOfBoundsAccess,
    ThreadRef,
    to_float32,
    values_sha256,
)
from ai_journey.kernel_launch_config import (
    Dim3,
    Index3,
    KernelLaunchError,
    LaunchConfig,
)

# Two blocks of four threads: linear thread IDs 0..7.
CONFIG = LaunchConfig(Dim3(2), Dim3(4))


def ref(thread_id):
    return ThreadRef(*CONFIG.coordinates(thread_id))


def run_as(recorder, thread_id, action):
    recorder.enter(thread_id)
    try:
        return action()
    finally:
        recorder.leave()


def test_float32_rounding_is_nearest_with_ties_to_even():
    assert to_float32(1.0 + 2**-24) == 1.0
    assert to_float32(1.0 + 3 * 2**-24) == 1.0 + 2**-22
    assert to_float32(0.1) == struct.unpack("<f", struct.pack("<f", 0.1))[0]
    assert to_float32(0.1) != 0.1
    assert math.copysign(1.0, to_float32(-0.0)) == -1.0
    for bad in (float("inf"), float("nan"), 1, True):
        with pytest.raises(KernelLaunchError, match="finite float"):
            to_float32(bad)
    with pytest.raises(KernelLaunchError, match="outside the float32 range"):
        to_float32(3.5e38)


@pytest.mark.parametrize(
    ("name", "dtype", "values", "message"),
    [
        ("Out", "uint8", [0], "lowercase identifier"),
        ("out", "int32", [0], "dtype must be one of"),
        ("out", "uint8", [], "1 to"),
        ("out", "uint8", [256], "cannot store 256"),
        ("out", "uint8", [True], "cannot store True"),
        ("out", "uint8", [1.0], "cannot store 1.0"),
        ("out", "float32", [0.1], "round it with to_float32"),
        ("out", "float32", [1], "cannot store 1"),
        ("out", "float32", [float("nan")], "cannot store nan"),
        ("out", "float32", [1e39], r"cannot store 1e\+39"),
    ],
)
def test_buffers_store_only_values_of_their_dtype(name, dtype, values, message):
    with pytest.raises(KernelLaunchError, match=message):
        DeviceBuffer(name, dtype, values)


def test_host_side_copies_and_digests_need_no_launch():
    gray = DeviceBuffer("gray", "uint8", [3, 1, 4])
    out = DeviceBuffer.zeros("out", "float32", 3)

    assert (gray.name, gray.dtype, gray.size) == ("gray", "uint8", 3)
    assert gray.to_host() == (3, 1, 4)
    assert out.to_host() == (0.0, 0.0, 0.0)
    assert gray.sha256() == sha256(bytes([3, 1, 4])).hexdigest()
    assert out.sha256() == sha256(struct.pack("<3f", 0.0, 0.0, 0.0)).hexdigest()
    assert values_sha256("float32", [0.5, -2.0]) == (
        sha256(struct.pack("<2f", 0.5, -2.0)).hexdigest()
    )
    with pytest.raises(KernelLaunchError, match="round it with to_float32"):
        values_sha256("float32", [0.1])
    with pytest.raises(KernelLaunchError, match="size must be between 1"):
        DeviceBuffer.zeros("out", "uint8", MAX_BUFFER_ELEMENTS + 1)


def test_kernels_cannot_bypass_the_recorder():
    data = DeviceBuffer("data", "uint8", [1, 2])

    with pytest.raises(KernelLaunchError, match="outside a kernel launch"):
        data[0]
    with pytest.raises(KernelLaunchError, match="outside a kernel launch"):
        data[0] = 1
    with pytest.raises(TypeError, match="not iterable"):
        list(data)
    with pytest.raises(TypeError):
        len(data)

    recorder = AccessRecorder(CONFIG, [data])
    with pytest.raises(KernelLaunchError, match="only by a running thread"):
        data[0]
    with pytest.raises(KernelLaunchError, match="host copies wait"):
        data.to_host()
    for index in (True, 1.0, slice(0, 1)):
        with pytest.raises(TypeError, match="index must be an integer"):
            run_as(recorder, 0, lambda index=index: data[index])
    recorder.finish()
    assert data.to_host() == (1, 2)


def test_out_of_bounds_accesses_are_recorded_and_never_touch_memory():
    data = DeviceBuffer("data", "uint8", [7, 8, 9])
    recorder = AccessRecorder(CONFIG, [data])

    assert run_as(recorder, 5, lambda: data[3]) == 0
    assert run_as(recorder, 6, lambda: data[-1]) == 0

    def write():
        data[10] = 200

    run_as(recorder, 7, write)
    assert run_as(recorder, 1, lambda: data[2]) == 9
    log = recorder.finish()

    assert data.to_host() == (7, 8, 9)
    assert log.out_of_bounds == (
        OutOfBoundsAccess("data", "read", 3, Index3(1), Index3(1)),
        OutOfBoundsAccess("data", "read", -1, Index3(1), Index3(2)),
        OutOfBoundsAccess("data", "write", 10, Index3(1), Index3(3)),
    )
    assert (log.out_of_bounds_reads, log.out_of_bounds_writes) == (2, 1)
    assert (log.reads, log.writes) == (1, 0)
    assert not log.memory_safe
    assert log.race_free
    assert log.out_of_bounds[2].to_mapping() == {
        "buffer": "data",
        "access": "write",
        "index": 10,
        "block": [1, 0, 0],
        "thread": [3, 0, 0],
    }


def test_canonical_records_do_not_depend_on_thread_order():
    def records(order):
        data = DeviceBuffer("data", "uint8", [0])
        recorder = AccessRecorder(CONFIG, [data])
        for thread_id in order:

            def touch(thread_id=thread_id):
                data[thread_id + 1] = 1
                data[thread_id + 10]

            run_as(recorder, thread_id, touch)
        return recorder.finish()

    forward = records(range(8))
    backward = records(reversed(range(8)))

    assert forward.out_of_bounds != backward.out_of_bounds
    assert forward.canonical_out_of_bounds == backward.canonical_out_of_bounds
    assert forward.canonical_out_of_bounds == forward.out_of_bounds
    assert [record.access for record in forward.canonical_out_of_bounds[:2]] == [
        "write",
        "read",
    ]


def test_two_writers_of_one_element_are_a_write_conflict_in_any_order():
    for order in ((6, 2), (2, 6)):
        out = DeviceBuffer("out", "uint8", [0, 0])
        recorder = AccessRecorder(CONFIG, [out])
        for thread_id in order:

            def store(thread_id=thread_id):
                out[1] = thread_id
                out[1] = thread_id

            run_as(recorder, thread_id, store)
        log = recorder.finish()

        assert len(log.write_conflicts) == 1
        conflict = log.write_conflicts[0]
        assert (conflict.buffer, conflict.index) == ("out", 1)
        assert conflict.writers == (ref(2), ref(6))
        assert conflict.to_mapping()["writers"][1] == {
            "block": [1, 0, 0],
            "thread": [2, 0, 0],
        }
        assert log.writes == 4
        assert log.writer_threads == {"out": 2}
        assert out.to_host()[1] == order[-1]
        assert not log.race_free


def test_one_thread_rewriting_its_own_element_is_not_a_race():
    out = DeviceBuffer("out", "uint8", [0])
    recorder = AccessRecorder(CONFIG, [out])

    def update():
        out[0] = 1
        out[0] = out[0] + 1

    run_as(recorder, 3, update)
    log = recorder.finish()

    assert out.to_host() == (2,)
    assert log.race_free
    assert log.counts() == {
        "reads": 1,
        "writes": 2,
        "out_of_bounds_reads": 0,
        "out_of_bounds_writes": 0,
        "write_conflicts": 0,
        "read_write_conflicts": 0,
    }


def test_a_read_of_another_threads_write_is_a_read_write_conflict():
    for order in ((0, 4), (4, 0)):
        shared = DeviceBuffer("shared", "uint8", [5, 6])
        recorder = AccessRecorder(CONFIG, [shared])

        def write():
            shared[0] = 1

        def read():
            return shared[0] + shared[1]

        for thread_id in order:
            run_as(recorder, thread_id, write if thread_id == 0 else read)
        log = recorder.finish()

        assert log.write_conflicts == ()
        assert len(log.read_write_conflicts) == 1
        conflict = log.read_write_conflicts[0]
        assert (conflict.buffer, conflict.index) == ("shared", 0)
        assert (conflict.writers, conflict.readers) == ((ref(0),), (ref(4),))
        assert set(conflict.to_mapping()) == {"buffer", "index", "writers", "readers"}


def test_reads_shared_by_many_threads_are_not_races():
    table = DeviceBuffer("table", "uint8", [9])
    out = DeviceBuffer("out", "uint8", [0] * 8)
    recorder = AccessRecorder(CONFIG, [table, out])
    for thread_id in range(8):

        def copy(thread_id=thread_id):
            out[thread_id] = table[0]

        run_as(recorder, thread_id, copy)
    log = recorder.finish()

    assert log.race_free and log.memory_safe
    assert log.writer_threads == {"out": 8, "table": 0}
    assert out.to_host() == (9,) * 8


def test_recorder_lifecycle_is_strict():
    first = DeviceBuffer("data", "uint8", [0])
    twin = DeviceBuffer("data", "uint8", [0])

    with pytest.raises(KernelLaunchError, match="distinct names"):
        AccessRecorder(CONFIG, [first, twin])
    with pytest.raises(TypeError, match="LaunchConfig"):
        AccessRecorder("config", [first])
    with pytest.raises(TypeError, match="DeviceBuffer"):
        AccessRecorder(CONFIG, [[0]])

    recorder = AccessRecorder(CONFIG, [first, first])
    with pytest.raises(KernelLaunchError, match="in a running launch"):
        AccessRecorder(CONFIG, [first])
    with pytest.raises(KernelLaunchError, match="thread_id must be between 0 and 7"):
        recorder.enter(8)
    recorder.enter(0)
    with pytest.raises(KernelLaunchError, match="still running"):
        recorder.enter(1)
    recorder.leave()
    recorder.finish()
    with pytest.raises(KernelLaunchError, match="finished"):
        recorder.finish()
    with pytest.raises(KernelLaunchError, match="finished"):
        recorder.enter(0)
    AccessRecorder(CONFIG, [first]).finish()
