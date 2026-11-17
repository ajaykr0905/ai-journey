import functools
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

from ai_journey.gpu_ceiling_reference import (
    bank_lane_sums,
    copy_source,
    fma_chain_values,
    fma_lane_words,
    fma_start_values,
    lane_tile,
    read_reduce_partials,
    reduce_source,
    stream_threads,
)
from ai_journey.gpu_ceiling_runtime import (
    STREAM_GUARD_FLOAT4,
    CeilingRuntime,
    GpuCeilingUnavailableError,
)
from ai_journey.gpu_ceiling_source import (
    BANK,
    FMA_CHAIN_COUNTS,
    KERNELS,
    GpuCeilingError,
    bank_tile_bytes,
)
from ai_journey.kernel_metal_geometry import (
    CheckedLaunch,
    PipelineLimits,
    image_launch,
    vector_launch,
)
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def ceiling_runtime():
    """Compile the kernels once per test process; call only under requires_metal."""

    return CeilingRuntime()


def launch(runtime, name, threads, group=256):
    return CheckedLaunch(vector_launch(threads, group), runtime.limits[name])


def sentinel_words(buffer):
    return set(buffer.view(numpy.uint32).tolist()) == {FLOAT32_SENTINEL_BITS}


@pytest.mark.parametrize(
    ("system", "module", "reason"),
    [
        (lambda: "Linux", AssertionError("torch must not be imported"), None),
        (darwin, ImportError("no torch"), "torch_not_importable"),
        (darwin, fake_torch(available=False), "mps_unavailable"),
        (darwin, fake_torch(compile_shader=False), "compile_shader_unavailable"),
    ],
)
def test_the_runtime_names_the_missing_requirement_from_the_metal_probe(
    system, module, reason
):
    expected = reason or "unsupported_operating_system"

    with pytest.raises(GpuCeilingUnavailableError, match=f"\\({expected}\\)"):
        CeilingRuntime(system=system, import_module=importer(module))
    assert issubclass(GpuCeilingUnavailableError, GpuCeilingError)


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.gpu_ceiling_runtime import CeilingRuntime\n"
        "from ai_journey.gpu_ceiling_runtime import GpuCeilingUnavailableError\n"
        "try:\n"
        "    CeilingRuntime()\n"
        "except GpuCeilingUnavailableError as error:\n"
        "    print(error)\n"
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
    on_macos = sys.platform == "darwin"
    reason = "torch_not_importable" if on_macos else "unsupported_operating_system"
    assert f"({reason})" in completed.stdout


@requires_metal
def test_every_pipeline_reports_its_limits_and_threadgroup_memory():
    runtime = ceiling_runtime()

    assert set(runtime.limits) == set(KERNELS)
    for name, limits in runtime.limits.items():
        assert isinstance(limits, PipelineLimits)
        assert limits.max_threads_per_threadgroup >= 256, name
        if KERNELS[name].family == BANK:
            row_words = int(name.rsplit("_", 1)[1])
            assert limits.static_threadgroup_memory_bytes == bank_tile_bytes(row_words)


@requires_metal
def test_each_fma_kernel_matches_the_exact_cpu_chains_and_keeps_its_guard():
    runtime = ceiling_runtime()
    start = fma_start_values(74)
    (values,) = fma_chain_values(start, (64,))

    for chains in FMA_CHAIN_COUNTS:
        name = f"fma_chains_{chains}"
        buffer = runtime.fma(chains, start, launch(runtime, name, 1000), 64)
        expected = lane_tile(fma_lane_words(values, chains), 1000)
        assert buffer.shape == (1024,)
        assert numpy.array_equal(buffer[:1000], expected), chains
        assert sentinel_words(buffer[1000:]), chains


@requires_metal
def test_the_stream_kernels_copy_and_sum_exactly_and_keep_their_guards():
    runtime = ceiling_runtime()
    count = 4099
    source = copy_source(74, 0, count)
    integers = reduce_source(74, 0, count)

    walkers = stream_threads(count, 4)
    copied = runtime.stream(
        "stream_copy",
        source,
        launch(runtime, "stream_copy", walkers),
        float4_count=count,
        per_thread=4,
    )
    assert copied.shape == (4 * (count + STREAM_GUARD_FLOAT4),)
    assert numpy.array_equal(copied[: 4 * count].view(numpy.uint32), source.view("u4"))
    assert sentinel_words(copied[4 * count :])

    walkers = stream_threads(count, 16)
    partials = runtime.stream(
        "stream_read_reduce",
        integers,
        launch(runtime, "stream_read_reduce", walkers),
        float4_count=count,
        per_thread=16,
    )
    assert partials.shape == (512,)
    assert numpy.array_equal(partials[:walkers], read_reduce_partials(integers, 257))
    assert sentinel_words(partials[walkers:])


@requires_metal
@pytest.mark.parametrize("row_words", [32, 33])
def test_the_bank_kernels_return_the_lane_sums_and_keep_their_guard(row_words):
    runtime = ceiling_runtime()
    name = f"bank_rows_{row_words}"

    for stride in (1, 3, 32):
        buffer = runtime.bank(row_words, stride, 70, launch(runtime, name, 1000))
        expected = lane_tile(bank_lane_sums(stride, row_words, 70), 1000)
        assert numpy.array_equal(buffer[:1000], expected), stride
        assert sentinel_words(buffer[1000:]), stride


@requires_metal
def test_workloads_bind_one_point_and_return_its_latest_output():
    runtime = ceiling_runtime()
    start = fma_start_values(74)
    (values,) = fma_chain_values(start, (16,))
    source = copy_source(74, 1, 1024)

    fma = runtime.fma_workload(4, start, launch(runtime, "fma_chains_4", 512), 16)
    bank = runtime.bank_workload(33, 8, 5, launch(runtime, "bank_rows_33", 512))
    device_source = runtime.upload(source)
    copy = runtime.stream_workload(
        "stream_copy",
        device_source,
        launch(runtime, "stream_copy", 512),
        float4_count=512,
        per_thread=1,
    )
    with pytest.raises(GpuCeilingError, match="ceil\\(count / per_thread\\)"):
        runtime.stream_workload(
            "stream_copy",
            device_source,
            launch(runtime, "stream_copy", 512),
            float4_count=512,
            per_thread=2,
        )
    for workload in (fma, bank, copy):
        workload.run()
        workload.run()
        workload.synchronize()
    assert numpy.array_equal(fma.output(), lane_tile(fma_lane_words(values, 4), 512))
    assert numpy.array_equal(bank.output(), lane_tile(bank_lane_sums(8, 33, 5), 512))
    assert numpy.array_equal(copy.output(), source[: 4 * 512])


@requires_metal
def test_launches_must_suit_the_kernel_that_runs_them():
    runtime = ceiling_runtime()
    start = fma_start_values(74)
    other = PipelineLimits(512, 32, 0)

    for wrong in (
        CheckedLaunch(vector_launch(64, 64), other),
        CheckedLaunch(image_launch(8, 8, (8, 8)), runtime.limits["fma_chains_1"]),
    ):
        with pytest.raises(GpuCeilingError, match="checked against its own pipeline"):
            runtime.fma(1, start, wrong, 8)
    with pytest.raises(GpuCeilingError, match="multiple of 8"):
        runtime.fma(1, start, launch(runtime, "fma_chains_1", 64), 12)
    with pytest.raises(GpuCeilingError, match="multiple of 32"):
        runtime.bank(32, 1, 1, launch(runtime, "bank_rows_32", 64, group=48))
    with pytest.raises(GpuCeilingError, match="stride must be in"):
        runtime.bank(32, 33, 1, launch(runtime, "bank_rows_32", 64))
