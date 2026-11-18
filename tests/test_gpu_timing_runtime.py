import functools
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_journey.gpu_timing_clocks import GpuTimingError
from ai_journey.gpu_timing_runtime import GpuTimingRuntime, GpuTimingUnavailableError
from ai_journey.gpu_timing_source import EMPTY_KERNEL, empty_launch
from ai_journey.kernel_metal_geometry import (
    CheckedLaunch,
    PipelineLimits,
    vector_launch,
)
from ai_journey.matmul_metal_mapping import matmul_shape
from ai_journey.matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from ai_journey.matmul_metal_runtime import SIDES
from ai_journey.matmul_metal_source import KERNELS
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal
from test_matmul_metal_runtime import launches

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def timing_runtime():
    """Compile the kernels once per test process; call only under requires_metal."""

    return GpuTimingRuntime()


def fake_kernel():
    return SimpleNamespace(
        max_threads_per_threadgroup=1024,
        thread_execution_width=32,
        static_thread_group_memory_length=0,
    )


@pytest.mark.parametrize(
    ("system", "module", "reason"),
    [
        (lambda: "Linux", AssertionError("torch must not be imported"), "operating"),
        (darwin, ImportError("no torch"), "torch_not_importable"),
        (darwin, fake_torch(available=False), "mps_unavailable"),
    ],
)
def test_a_host_without_metal_is_unavailable_not_a_crash(system, module, reason):
    with pytest.raises(GpuTimingUnavailableError, match=reason):
        GpuTimingRuntime(system=system, import_module=importer(module))


def test_a_rejected_empty_kernel_is_an_unavailable_host():
    def compile_shader(source):
        if EMPTY_KERNEL in source:
            raise RuntimeError("program_source:4: error: expected ';'")
        return SimpleNamespace(**{name: fake_kernel() for name in KERNELS})

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=compile_shader)
    torch.get_float32_matmul_precision = lambda: "highest"

    with pytest.raises(GpuTimingUnavailableError, match="rejected the empty kernel"):
        GpuTimingRuntime(system=darwin, import_module=importer(torch))


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.gpu_timing_runtime import GpuTimingRuntime\n"
        "try:\n"
        "    GpuTimingRuntime()\n"
        "except Exception as error:\n"
        "    print(type(error).__name__, error)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("GpuTimingUnavailableError ")


@requires_metal
def test_the_runtime_adds_the_empty_pipeline_to_the_day_74_limits():
    runtime = timing_runtime()

    assert set(runtime.limits) == {*KERNELS, EMPTY_KERNEL}
    assert runtime.backend == "torch.mps.compile_shader"
    assert runtime.limits[EMPTY_KERNEL].max_threads_per_threadgroup >= 1


@requires_metal
def test_the_empty_side_runs_only_with_its_own_checked_launch():
    runtime = timing_runtime()
    launch = CheckedLaunch(empty_launch(), runtime.limits[EMPTY_KERNEL])

    side = runtime.empty_side(launch)
    side()
    runtime.synchronize()

    limits = runtime.limits[EMPTY_KERNEL]
    other = PipelineLimits(
        max_threads_per_threadgroup=limits.max_threads_per_threadgroup,
        thread_execution_width=limits.thread_execution_width,
        static_threadgroup_memory_bytes=limits.static_threadgroup_memory_bytes + 16,
    )
    for wrong in (
        CheckedLaunch(vector_launch(2, 1), limits),
        CheckedLaunch(empty_launch(), other),
        empty_launch(),
    ):
        with pytest.raises(GpuTimingError, match="one-thread launch"):
            runtime.empty_side(wrong)


@requires_metal
def test_an_event_pair_times_one_launch_as_positive_milliseconds():
    runtime = timing_runtime()
    side = runtime.empty_side(
        CheckedLaunch(empty_launch(), runtime.limits[EMPTY_KERNEL])
    )
    pair = runtime.event_pair()

    runtime.synchronize()
    pair.start()
    side()
    pair.stop()
    runtime.synchronize()
    elapsed = pair.elapsed_ms()

    assert type(elapsed) is float
    assert math.isfinite(elapsed) and elapsed > 0.0


@requires_metal
def test_the_workload_binds_the_three_day_74_sides():
    runtime = timing_runtime()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(75, BENCHMARK_STREAM, 0, shape)

    workload = runtime.workload(left, right, launches(runtime, shape))
    for call in workload.sides.values():
        call()
    check = workload.check()

    assert tuple(workload.sides) == SIDES
    assert check.naive_same_bits is True


@requires_metal
def test_an_event_interval_holds_the_host_time_between_its_records():
    # The guide says the event clock is not GPU time alone; this is the proof.
    runtime = timing_runtime()
    side = runtime.empty_side(
        CheckedLaunch(empty_launch(), runtime.limits[EMPTY_KERNEL])
    )
    pair = runtime.event_pair()

    runtime.synchronize()
    pair.start()
    time.sleep(0.02)
    side()
    pair.stop()
    runtime.synchronize()

    assert pair.elapsed_ms() >= 15.0


@requires_metal
def test_an_event_pair_brackets_the_gpu_work_of_a_heavy_launch():
    # x_rows at 1024 cubed takes about 11 ms here; an end event that did not
    # wait for the kernel would read near the 0.07 ms of the empty kernel.
    runtime = timing_runtime()
    shape = matmul_shape(1024, 1024, 1024)
    left, right = matmul_inputs(75, BENCHMARK_STREAM, 0, shape)
    workload = runtime.workload(left, right, launches(runtime, shape))
    pair = runtime.event_pair()

    workload.synchronize()
    pair.start()
    workload.sides["matmul_x_rows"]()
    pair.stop()
    workload.synchronize()

    assert pair.elapsed_ms() >= 1.0
