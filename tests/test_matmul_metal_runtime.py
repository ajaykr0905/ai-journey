import functools
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS
from ai_journey.matmul_metal_benchmark import agreement_bound
from ai_journey.matmul_metal_mapping import (
    X_COLUMNS,
    X_ROWS,
    MatmulMetalError,
    matmul_launch,
    matmul_shape,
)
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    error_bound,
    float64_reference,
    fma_replay,
    matmul_inputs,
    max_relative_error,
)
from ai_journey.matmul_metal_runtime import (
    SIDES,
    TORCH_MATMUL,
    MatmulMetalRuntime,
    MatmulMetalUnavailableError,
)
from ai_journey.matmul_metal_source import KERNEL_FOR_MAPPING, KERNELS
from test_kernel_metal_runtime import darwin, fake_torch, importer, requires_metal
from test_matmul_metal_reference import TRAPS, trap_operands

ROOT = Path(__file__).resolve().parents[1]
GROUP = (32, 8)


@functools.cache
def matmul_runtime():
    """Compile the kernels once per test process; call only under requires_metal."""

    return MatmulMetalRuntime()


def launches(runtime, shape, group=GROUP):
    return {
        KERNEL_FOR_MAPPING[mapping]: CheckedLaunch(
            matmul_launch(mapping, shape, group),
            runtime.limits[KERNEL_FOR_MAPPING[mapping]],
        )
        for mapping in (X_COLUMNS, X_ROWS)
    }


@pytest.mark.parametrize(
    ("module", "reason"),
    [
        (ImportError("no torch"), "torch_not_importable"),
        (fake_torch(built=False), "pytorch_without_mps"),
        (fake_torch(available=False), "mps_unavailable"),
        (fake_torch(compile_shader=False), "compile_shader_unavailable"),
    ],
)
def test_a_host_without_the_requirements_is_named_before_compiling(module, reason):
    with pytest.raises(MatmulMetalUnavailableError, match=reason):
        MatmulMetalRuntime(system=darwin, import_module=importer(module))
    with pytest.raises(MatmulMetalUnavailableError, match="unsupported_operating"):
        MatmulMetalRuntime(
            system=lambda: "Linux",
            import_module=importer(AssertionError("torch must not be imported")),
        )


def test_a_rejected_source_is_an_unavailable_host_not_a_crash():
    def refuse(source):
        assert "kernel void matmul_x_rows(" in source
        raise RuntimeError("program_source:9: error: unknown type name")

    torch = fake_torch()
    torch.mps = SimpleNamespace(compile_shader=refuse)

    with pytest.raises(MatmulMetalUnavailableError, match="rejected the matmul source"):
        MatmulMetalRuntime(system=darwin, import_module=importer(torch))


def test_importing_the_runtime_without_torch_never_loads_it():
    script = (
        "import sys; sys.modules['torch'] = None\n"
        "from ai_journey.matmul_metal_runtime import MatmulMetalRuntime\n"
        "try:\n"
        "    MatmulMetalRuntime()\n"
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
    reason = (
        "torch_not_importable"
        if sys.platform == "darwin"
        else "unsupported_operating_system"
    )
    assert completed.stdout.startswith("MatmulMetalUnavailableError ")
    assert f"({reason})" in completed.stdout


@requires_metal
def test_both_pipelines_report_usable_limits():
    runtime = matmul_runtime()

    assert set(runtime.limits) == set(KERNELS)
    for limits in runtime.limits.values():
        assert isinstance(limits, PipelineLimits)
        assert limits.max_threads_per_threadgroup >= GROUP[0] * GROUP[1]
    assert runtime.float32_matmul_precision in ("highest", "high", "medium")
    assert SIDES == ("matmul_x_columns", "matmul_x_rows", TORCH_MATMUL)


@requires_metal
@pytest.mark.parametrize(("start", "a", "b", "fused", "twice_rounded"), TRAPS)
def test_the_gpu_rounds_each_fma_step_once(start, a, b, fused, twice_rounded):
    runtime = matmul_runtime()
    left, right = trap_operands(start, a, b)
    shape = matmul_shape(1, 1, 2)

    for name, launch in launches(runtime, shape).items():
        buffer = runtime.run(name, left, right, launch)
        assert buffer[0] == numpy.float32(fused), name


@requires_metal
@pytest.mark.parametrize(("m", "n", "k"), [(17, 33, 9), (33, 17, 65), (64, 8, 300)])
def test_ragged_outputs_replay_exactly_and_keep_the_guard_tail(m, n, k):
    runtime = matmul_runtime()
    shape = matmul_shape(m, n, k)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, shape)
    replay = fma_replay(left, right).reshape(-1).view(numpy.uint32)

    for name, launch in launches(runtime, shape).items():
        buffer = runtime.run(name, left, right, launch)
        guard = launch.geometry.padding_threads
        assert buffer.shape == (m * n + guard,), name
        assert numpy.array_equal(buffer[: m * n].view(numpy.uint32), replay), name
        assert set(buffer[m * n :].view(numpy.uint32).tolist()) <= {
            FLOAT32_SENTINEL_BITS
        }
    reference, magnitude = float64_reference(left, right)
    for device in ("mps", "cpu"):
        product = runtime.torch_matmul(left, right, device)
        assert product.shape == (m, n)
        assert max_relative_error(product, reference, magnitude) <= error_bound(k)


@requires_metal
def test_a_workload_times_every_side_and_checks_their_agreement():
    runtime = matmul_runtime()
    shape = matmul_shape(65, 47, 129)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 1, shape)

    workload = runtime.workload(left, right, launches(runtime, shape))
    for name in SIDES:
        workload.sides[name]()
    workload.synchronize()
    check = workload.check()

    assert list(workload.sides) == list(SIDES)
    assert check.naive_same_bits is True
    assert 0.0 <= check.torch_disagreement <= 2 * error_bound(129)


@requires_metal
def test_kernel_calls_pass_whole_groups_and_32_bit_scalars():
    # Without arg_casts a little-endian GPU still reads the low 32 bits, and a
    # grid of exactly the extent leaves the outputs right, so only the bound
    # call itself shows the documented launch.
    runtime = matmul_runtime()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, shape)
    checked = launches(runtime, shape)

    workload = runtime.workload(left, right, checked)

    for name, launch in checked.items():
        call = workload.sides[name]
        assert call.args[3:] == (17, 33, 9), name
        assert call.keywords == {
            "threads": list(launch.geometry.threads),
            "group_size": [32, 8],
            "arg_casts": {3: "int32", 4: "int32", 5: "int32"},
        }, name


@requires_metal
def test_the_runtime_reads_the_float32_matmul_precision_in_force():
    import torch

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium")
        runtime = MatmulMetalRuntime()
    finally:
        torch.set_float32_matmul_precision(previous)

    assert runtime.float32_matmul_precision == "medium"


@requires_metal
def test_the_workload_check_sees_outputs_that_moved():
    # The check is the only guard on the timed outputs, so it must notice a
    # torch.matmul output that moved and naive outputs that differ in a bit.
    import torch

    runtime = matmul_runtime()
    shape = matmul_shape(65, 47, 129)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 1, shape)
    workload = runtime.workload(left, right, launches(runtime, shape))
    for name in SIDES:
        workload.sides[name]()
    rows = workload.sides["matmul_x_rows"].args[0]
    product = workload.sides[TORCH_MATMUL].keywords["out"]

    product[3, 5] += 1.0
    moved_torch = workload.check()
    rows.view(torch.int32)[0, 0] += 1
    moved_rows = workload.check()

    assert moved_torch.naive_same_bits is True
    assert moved_torch.torch_disagreement > agreement_bound(shape.k)
    assert moved_rows.naive_same_bits is False


@requires_metal
def test_launches_must_match_the_kernels_pipeline_and_mapping():
    runtime = matmul_runtime()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, shape)
    good = launches(runtime, shape)
    other = PipelineLimits(512, 32, 0)

    for name, launch in (
        ("matmul_x_columns", good["matmul_x_rows"]),
        ("matmul_x_rows", CheckedLaunch(matmul_launch(X_ROWS, shape, GROUP), other)),
        ("matmul_x_rows", matmul_launch(X_ROWS, shape, GROUP)),
    ):
        with pytest.raises(MatmulMetalError, match="checked against its own pipeline"):
            runtime.run(name, left, right, launch)
    with pytest.raises(MatmulMetalError, match="kernel must be one of"):
        runtime.run("matmul_tiled", left, right, good["matmul_x_rows"])
    with pytest.raises(MatmulMetalError, match="conformable 2D float32"):
        runtime.run("matmul_x_rows", left.T, right, good["matmul_x_rows"])
    with pytest.raises(MatmulMetalError, match="device must be one of"):
        runtime.torch_matmul(left, right, "cuda")
    with pytest.raises(MatmulMetalError, match="one launch per naive kernel"):
        runtime.workload(left, right, {"matmul_x_rows": good["matmul_x_rows"]})
